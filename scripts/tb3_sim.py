import time
import threading
from queue import Queue

import mujoco as mj
import cv2
import numpy as np


import os
import sys

# 현재 파일(tb3_sim.py)의 절대 경로를 가져옵니다.
current_file = os.path.abspath(__file__) 
# scripts 폴더의 경로
scripts_dir = os.path.dirname(current_file) 
# 프로젝트 루트 (/home/arcturus/mujoco_ws/src/mujoco_llm)
project_root = os.path.dirname(scripts_dir) 

# 프로젝트 루트를 파이썬 경로 최상단에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 이제 utils를 안전하게 불러올 수 있습니다.
from utils.lidar import Lidar
from utils.mujoco_renderer import MuJoCoViewer
from utils.object_detector import ObjectDetector



ACTION_TABLE = {
    "멈춤": (0.0, 0.0),
    "직진": (8.0, 8.0),
    "후진": (-8.0, -8.0),
    "좌회전": (6.0, 8.0),
    "우회전": (8.0, 6.0),
    "제자리 회전": (4.0, -4.0),
}

# YOLO 라벨 ↔ MuJoCo body 이름 매핑
XML_BODY_MAP = {
    "go_straight":  "card_go_straight",
    "rotate":       "card_rotate",
    "turn_left":    "card_turn_left",
    "turn_right":   "card_turn_right",
}

# 카드용 색상(BGR) 매핑
# 이전에는 plot을 사용해 색 지정이 필요 없었음
# 신뢰도와 거리를 확인하기 위해 사용자 지정 텍스트(cv2)를 사용하기 때문에 색 구분 필요
CARD_COLOR_MAP = {
    "go_straight":  (0, 255, 0),    # 직진: 초록색
    "turn_left":    (255, 255, 0),  # 좌회전: 하늘색
    "turn_right":   (0, 165, 255),  # 우회전: 주황색
    "rotate":       (0, 0, 255)     # 회전: 빨강색    
}


class TurtlebotFactorySim:
    """
    MuJoCo 기반 터틀봇3 팩토리 시뮬 통합 클래스.

    기능:
    - tb3_factory_cards.xml 로드
    - 메인뷰 + 로봇 카메라 렌더링
    - latest_frame 에 로봇 카메라 마지막 프레임(BGR) 저장
    - (옵션) YOLO로 로봇 카메라 프레임 감지 & cv2 창으로 출력
    - (옵션) command_queue 에서 명령을 읽어와 apply_command()로 처리
    """

    def __init__(
        self,
        xml_path: str | None = None,
        use_yolo: bool = False,
        yolo_weight_path: str | None = None,
        yolo_conf: float = 0.5,
        command_queue: Queue | None = None,
        fps: int = 60,
        current_action = None,
        action_end_sim_time = 0.0,
    ):
        
        PROJECT_ROOT = os.path.abspath(os.getcwd())

        if xml_path is None:
            xml_path = os.path.join(
                PROJECT_ROOT,
                "asset",
                "robotis_tb3",
                "tb3_factory_cards.xml",
            )

        print(f"[TurtlebotFactorySim] Loading scene from: {xml_path}")

        # 검색 모드 타겟 레이블
        self.search_target_label = None  
        self.current_action = current_action
        self.action_end_sim_time = action_end_sim_time
        # ===== MuJoCo 모델/데이터 로드 =====
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # 기존 MuJoCoViewer 사용
        self.viewer = MuJoCoViewer(self.model, self.data)

        # ===== 카메라 프레임 저장용 =====
        # 항상 "로봇 카메라 기준 BGR 이미지"를 최신 상태로 보관
        self.latest_frame: np.ndarray | None = None

        # ===== YOLO 옵션 =====
        self.use_yolo = use_yolo
        self.detector = None
        self.yolo_window_name = "Robot YOLO View"

        if self.use_yolo:
            if yolo_weight_path is None:
                raise ValueError("use_yolo=True 인데 yolo_weight_path 가 없습니다.")
            if not os.path.exists(yolo_weight_path):
                raise FileNotFoundError(f"YOLO weight not found: {yolo_weight_path}")

            print(f"[TurtlebotFactorySim] Loading ObjectDetector: {yolo_weight_path}")
            self.detector = ObjectDetector(yolo_weight_path, conf=yolo_conf)

            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.yolo_window_name, 960, 720)
            cv2.moveWindow(self.yolo_window_name, 1280, 50)

        # ===== 명령 큐 (LLM / 키보드 등에서 넣어주는 명령) =====
        self.command_queue = command_queue if command_queue is not None else Queue()

        # ===== 루프 설정 =====
        self.fps = fps
        self._running = False

        # 로봇 base body 이름 (XML 기준)
        self.robot_body_name = "base"

        # 프레임 카운터
        self.render_counter = 0
        # main glfw 창의 distance값 캐싱
        self.cached_gt_info = ""



    # ------------------------------------------------------------------
    # 외부에서 사용할 수 있는 유틸 메서드들
    # ------------------------------------------------------------------
    def step_simulation(self):
        """한 타임스텝(fps 기준)만큼 시뮬레이션을 진행."""
        time_prev = self.data.time
        dt = 1.0 / self.fps
        while self.data.time - time_prev < dt:
            self.viewer.step_simulation()

    # 계산된 실제 거리(ground_truth, gt) Viewer에 전달
    def render(self):
        # 1. 카운터 증가
        self.render_counter += 1

        # 2. 10프레임마다 한 번씩만 텍스트 갱신 (약 60fps 기준 0.16초마다)
        if self.render_counter % 10 == 0 or self.cached_gt_info == "":
            gt_info_str = "Distance to Cards (GT) | Error:\n"
            for label in ["go_straight", "rotate", "turn_left", "turn_right"]:
                res = self.get_target_distance_and_time(label)
                dist = res[0] if res is not None else None
                real_dist = self.get_ground_truth_distance(label)
                
                error_rate = None
                if dist is not None and real_dist is not None and real_dist != 0:
                    error_rate = abs(dist - real_dist) / real_dist * 100.0

                if real_dist is not None:
                    dist_str = f"{dist:.2f}m" if dist is not None else "N/A"
                    err_str = f"({error_rate:.1f}%)" if error_rate is not None else "(N/A)"
                    gt_info_str += f"  {label:<12}: {dist_str:>7} {err_str:>8}\n"
                else:
                    gt_info_str += f"  {label:<12}: N/A\n"
            
            # 갱신된 문자열을 캐시에 저장
            self.cached_gt_info = gt_info_str
            self.render_counter = 0 # 카운터 초기화 (선택 사항)

        # 3. 실제 출력은 캐시된 데이터를 사용 (매 프레임 계산하지 않음)
        self.viewer.render_main(overlay_type="imu", extra_info=self.cached_gt_info)

        # 로봇 카메라 렌더링 및 캡처 (기존 로직)
        self.viewer.render_robot()
        if hasattr(self.viewer, "capture_img"):
            self.latest_frame = self.viewer.capture_img()

        self.viewer.poll_events()


    def apply_command(self, cmd: str, base_duration: float = 1.0):
        cmd = cmd.strip()

        # 1) 카드 검색 계열 액션 처리
        SEARCH_MAP = {
            "SEARCH_GO_STRAIGHT": "go_straight",
            "SEARCH_ROTATE":      "rotate",   
            "SEARCH_TURN_LEFT":   "turn_left",
            "SEARCH_TURN_RIGHT":  "turn_right",
        }

        if cmd in SEARCH_MAP:
            target = SEARCH_MAP[cmd]
            self.search_target_label = target

            # 제자리 회전 시작 (좌우 반대 방향으로)
            self.data.ctrl[0] = 4.0
            self.data.ctrl[1] = -4.0

            self.current_action = cmd
            # 검색 모드는 duration으로 멈추지 않게, action_end_sim_time은 무시
            self.action_end_sim_time = float("inf")

            print(f"[TurtlebotFactorySim] Start search for '{target}' (cmd={cmd})")
            return
        
        
        # 2) 일반 ACTION_TABLE 기반 액션 처리
        if cmd not in ACTION_TABLE:
            print(f"[TurtlebotFactorySim] Unknown command: {cmd}")
            return

        duration = base_duration
        if cmd in ["좌회전", "우회전"]:
            duration *= 1.6
        elif cmd == "제자리 회전":
            duration *= 1.0

        left, right = ACTION_TABLE[cmd]
        self.data.ctrl[0] = left
        self.data.ctrl[1] = right

        self.current_action = cmd
        self.action_end_sim_time = self.data.time + duration

        print(f"[TurtlebotFactorySim] Command '{cmd}' → L={left}, R={right}, duration={duration:.2f}s")

    def _process_commands(self):
        """command_queue 에 쌓인 명령들을 한 번에 처리."""
        while not self.command_queue.empty():
            cmd = self.command_queue.get()
            self.apply_command(cmd)

    def yolo_detect_dict(self):
        if (not self.use_yolo) or (self.detector is None) or (self.latest_frame is None):
            return {}
        return self.detector.detect_dict(self.latest_frame)

    def yolo_detect_image(self):
        if (not self.use_yolo) or (self.detector is None) or (self.latest_frame is None):
            return None
        return self.detector.detect_image(self.latest_frame)

    # YOLO 화면데 LiDAR 측정값  표
    def _run_yolo_on_latest_frame(self):
        if not self.use_yolo or self.detector is None or self.latest_frame is None:
            return

        # 원본 프레임 복사 (화면에 그리기 위)
        display_img = self.latest_frame.copy()

        # 감지 수
        det_dict = self.detector.detect_dict(display_img)
        
        for label, objects in det_dict.items():
            # lidar 거리 측
            # [수정] 무거운 LiDAR 연산은 루프당 딱 1번만 수행해서 변수에 저장
            res = self.get_target_distance_and_time(label)
            lidar_dist = res[0] if res else None
            
            # 색상 선택(CARD_COLOR_MAP)
            color = CARD_COLOR_MAP.get(label, (0, 255, 0))

            # 바운딩 박스 및 텍스트 그리기 
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                conf = obj['confidence']
                
                # 텍스트 정보 구성 (1줄로 줄여서 연산 최소화)
                # text = f"{label} {conf:.2f}"
                text = f"{label}"
                if lidar_dist is not None and lidar_dist < 10.0:
                    text += f" | {lidar_dist:.2f}m"
                
                # 디버깅 로그 출력
                # print(f"Dist : {lidar_dist:.2f}m, Real : {self.get_ground_truth_distance(label):.2f}m, Diff : {abs(lidar_dist - self.get_ground_truth_distance(label)):.2f}m")

                # 시각화 (박스와 글자만 간단히)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_img, text, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(self.yolo_window_name, display_img)


    # get_target_distance_and_time을 호출하여 거리만 반환
    def get_dist_to_target(self, target_label):
        dist, _ = self.get_target_distance_and_time(target_label)
        return dist

    # ======================================================= #
    # TODO: LIDAR로 측정한 거리와 실제로 측정한 거리가 동일하게 만들어보기!
    # ======================================================= #
    
    def get_target_distance_and_time(self, target_label: str):
        # YOLO 결과 가져오기
        det_dict = self.yolo_detect_dict()
        if target_label not in det_dict:
            return None, None 

        target_obj = det_dict[target_label][0] 
        cx, cy = target_obj["center"]
        x1, y1, x2, y2 = target_obj["bbox"]
        bbox_height = y2 - y1 
        
        # 1. LiDAR 데이터 가져오기 (사용자 XML의 'laser' 이름 기준 우선 시도)
        ranges = np.array([])
        # 센서 이름 우선순위: XML 구성에 따라 'laser', 'rf', 'scan' 순서
        for name in ["laser", "rf", "scan", "lidar"]:
            temp_ranges = Lidar.get_lidar_ranges(self.model, self.data, name)
            if len(temp_ranges) > 0:
                ranges = temp_ranges
                break

        # [실시간 출력] 0번 인덱스(정면) LiDAR 값 출력
        # if len(ranges) > 0:
        #     print(f">>> [Lidar Debug] Front(Index 0) Distance: {ranges[0]:.4f}m")
        # else:

        #     print(">>> [Lidar Debug] No Lidar data found. Check sensor name in XML.")

        final_dist = 15.0 # 초기값

        # 2. 타겟 거리 추출 로직 (get_target_distance_and_time 내부 수정)
        if len(ranges) >= 360:
            # 1. 인덱스 계산 (정밀도 유지를 위해 round 사용)
            target_angle_deg = (cx - 320) / 640 * 60.0
            lidar_idx = int(round(target_angle_deg)) % 360
            
            # 2. 검색 범위 확대 (가장자리 카드를 위해 +-8도까지 탐색)
            search_range = 8 
            indices = [(lidar_idx + i) % 360 for i in range(-search_range, search_range + 1)]
            
            # 3. 유효 거리 필터링 (0.2m ~ 3.5m 사이만 '카드'로 인정)
            # 0.2m 미만: 로봇 몸체에 반사된 노이즈 방지
            # 3.5m 초과: 저 멀리 있는 벽 데이터 방지
            valid_samples = [ranges[i] for i in indices if 0.2 < ranges[i] < 3.5]
            
            if valid_samples:
                # 카드 후보 중 가장 가까운 거리를 선택
                final_dist = np.min(valid_samples)
            else:
                # LiDAR가 놓쳤을 경우 Vision Fallback (bbox 높이 기반)
                if bbox_height > 0:
                    final_dist = 160.0 / bbox_height 
                else:
                    final_dist = 15.0

            # 디버깅 로그
            # print(f"Target: {target_label} | Lidar_Idx: {lidar_idx} | Dist: {final_dist:.2f}m")

        # 3. Vision Fallback 및 시간 계산 (기존 로직 유지)
        if final_dist >= 14.5 and bbox_height > 0:
            final_dist = 160.0 / bbox_height # Fallback

        target_gap = 0.10
        robot_speed = 0.22 
        time_to_stop = max(0.0, (final_dist - target_gap) / robot_speed)

        return final_dist, time_to_stop

    # 실제 거리(ground_truth, gt) 측정용 메서드
    def get_ground_truth_distance(self, target_label: str):
        # MuJoCo 시뮬레이션 내부 좌표를 사용하여 '실제 물리적 거리'를 계산
        try:
            # 1. 타겟 Body의 위치 가져오기
            body_name = XML_BODY_MAP.get(target_label, target_label)
            target_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
            if target_id == -1: return None
            target_pos = self.data.xpos[target_id]

            # 2. 로봇의 기준점 위치 가져오기
            try:
                robot_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
                robot_pos = self.data.xpos[robot_id]
            except:
                return None

            # 3. 2D 평면 거리(x, y) 계산
            dist = np.linalg.norm(target_pos[:2] - robot_pos[:2])
            return dist
        except:
            return None

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self.lidar_counter = 0
        self.current_dist = 15.0
        print("[TurtlebotFactorySim] Start simulation loop.")
        try:
            while self._running and not self.viewer.should_close():
                # 1) 명령 처리
                self._process_commands()

                # 2) 시뮬레이션 한 스텝
                self.step_simulation()

                # 3) 렌더 + latest_frame 갱신
                self.render()

                # --- [추가된 구간: LiDAR 연산 빈도 조절] ---
                # 접근 중(APPROACH)일 때만 거리 계산을 수행하되, 5번에 1번만 실행
                if self.current_action and self.current_action.startswith("APPROACH_"):
                    self.lidar_counter += 1
                    if self.lidar_counter % 5 == 0:
                        # 타겟 라벨은 ACTION 이름에서 추출하거나 저장된 변수 사용
                        target_label = self.current_action.split("_")[1] 
                        self.current_dist, _ = self.get_target_distance_and_time(target_label)
                        self.lidar_counter = 0
                        
                        # 거리가 10cm(target_gap) 이하면 정지 로직
                        if self.current_dist and self.current_dist <= 0.10:
                            self.data.ctrl[0] = 0.0
                            self.data.ctrl[1] = 0.0
                            print(f"[TurtlebotFactorySim] '{target_label}' 도달 완료.")
                            self.current_action = None
                # ----------------------------------------


                # 3.5) 검색 모드라면: YOLO로 타겟 감시
                if self.search_target_label and self.current_action.startswith("SEARCH_"):
                    det = self.yolo_detect_dict()
                    if self.search_target_label and (self.search_target_label in det):
                        # 타겟 발견 → 정지 + 검색 종료
                        self.data.ctrl[0] = 0.0
                        self.data.ctrl[1] = 0.0
                        print(f"[TurtlebotFactorySim] Found '{self.search_target_label}' → stop search.")
                        self.search_target_label = None
                        self.current_action = None
                        self.action_end_sim_time = 0.0

                # 4) 일반 액션 duration 기반 정지 (검색 모드일 땐 X)
                elif (
                    self.current_action 
                    and not (self.current_action.startswith("SEARCH_"))
                    and not (self.current_action.startswith("APPROACH_"))
                    and self.data.time > self.action_end_sim_time
                ):
                    self.data.ctrl[0] = 0.0
                    self.data.ctrl[1] = 0.0
                    print(f"[TurtlebotFactorySim] '{self.current_action}' 완료 → stop.")
                    self.current_action = None

                # 5) YOLO 디스플레이
                if self.use_yolo:
                    self._run_yolo_on_latest_frame()

                # 6) q로 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[TurtlebotFactorySim] 'q' 입력으로 종료합니다.")
                    break

        except Exception as e:
            print(f"\n[TurtlebotFactorySim] 시뮬레이션 중 예외 발생: {e}")
        finally:
            self.close()

    def close(self):
        """시뮬레이션 종료 및 리소스 정리."""
        self._running = False
        if self.use_yolo:
            cv2.destroyWindow(self.yolo_window_name)
        self.viewer.terminate()
        print("[TurtlebotFactorySim] Simulation terminated.")
