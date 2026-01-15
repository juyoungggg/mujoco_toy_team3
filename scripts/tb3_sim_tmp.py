import os
import sys
import time
import threading
from queue import Queue

import mujoco as mj
import cv2
import numpy as np

# 프로젝트 루트에서 utils 가져오기
"""
current_file_path = os.path.abspath(__file__)       # /mujoco_llm/scripts/tb3_sim.py
current_dir = os.path.dirname(current_file_path)    # /mujoco_llm/scripts
PROJECT_ROOT= os.path.dirname(current_dir)          # /mujoco_llm
"""

sys.path.append(os.path.abspath(os.getcwd()))

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
            cv2.resizeWindow(self.yolo_window_name, 640, 480)

        # ===== 명령 큐 (LLM / 키보드 등에서 넣어주는 명령) =====
        self.command_queue = command_queue if command_queue is not None else Queue()

        # ===== 루프 설정 =====
        self.fps = fps
        self._running = False

        # 로봇 base body 이름 (XML 기준)
        self.robot_body_name = "base"

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
        gt_info_str = "Distance to Cards (GT):\n"
        for label in ["go_straight", "rotate", "turn_left", "turn_right"]:
            dist = self.get_ground_truth_distance(label)
            if dist is not None:
                gt_info_str += f"  {label:<12}: {dist:.2f} m\n"
            else:
                gt_info_str += f"  {label:<12}: N/A\n"

        """메인뷰 + 로봇 카메라 렌더링, latest_frame 업데이트."""
        # 메인 뷰: IMU overlay
        self.viewer.render_main(overlay_type="imu", extra_info=gt_info_str)

        # 로봇 카메라 화면 표시 + 이미지 캡처
        self.viewer.render_robot()
        # MuJoCoViewer 안에 capture_img() 가 로봇 카메라 뷰를 BGR로 반환한다고 가정
        if hasattr(self.viewer, "capture_img"):
            frame_bgr = self.viewer.capture_img()
            self.latest_frame = frame_bgr
        else:
            self.latest_frame = None

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
        
        """
        # 카드 전방 10cm까지 이동 (TBA)
        APPROACH_MAP = {
            "APPROACH_GO_STRAIGHT": "go_straight",
            "APPROACH_ROTATE":      "rotate",
            "APPROACH_TURN_LEFT":   "turn_left",
            "APPROACH_TURN_RIGHT":  "turn_right",
        }
        if cmd in APPROACH_MAP:
            target = APPROACH_MAP[cmd]
            self.search_target_label = target # 타겟 설정
            
            # 천천히 직진 (속도 4.0)
            self.data.ctrl[0] = 4.0
            self.data.ctrl[1] = 4.0
            
            self.current_action = cmd
            self.action_end_sim_time = float("inf") # 센서로 멈출 거라 시간 무제한
            print(f"[Sim] Start approaching '{target}' (Stop at 10cm)")
            return
        """
        
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

    # YOLO 화면에 LiDAR 측정값 표시
    def _run_yolo_on_latest_frame(self):
        if not self.use_yolo or self.detector is None or self.latest_frame is None:
            return

        # 원본 프레임 복사 (화면에 그리기 위함)
        display_img = self.latest_frame.copy()
        
        # 감지 수행
        det_dict = self.detector.detect_dict(display_img)
        
        # 결과 그리기
        for label, objects in det_dict.items():
            # LiDAR 거리 측정
            lidar_dist, _ = self.get_target_distance_and_time(label)

            # 색상 선택 (CARD_COLOR_MAP)
            color = CARD_COLOR_MAP.get(label, (0, 255, 0))  # 없으면 초록색

            # 텍스트 구성
            info_text = f"{label}"
            if lidar_dist is not None:
                info_text += f" | {lidar_dist:.2f}m"

            # 바운딩 박스 및 텍스트 그리기
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                confidence = obj['confidence']

                full_text = f"{info_text} ({confidence:.2f})"

                # 박스
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

                # 텍스트 배경
                (w, h), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                # 텍스트
                cv2.putText(display_img, full_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
        bbox_height = y2 - y1 # 물체의 화면상 높이 (픽셀)
        
        lidar_dist = 15.0 # 기본값 (못 찾았을 경우)
        
        # 1. LiDAR 데이터 가져오기 (이름 'scan', 'laser', 'lidar' 순 시도)
        ranges = np.array([])
        for name in ["scan", "laser", "lidar", "lidar_front"]:
            temp_ranges = Lidar.get_lidar_ranges(self.model, self.data, name)
            if len(temp_ranges) > 0:
                ranges = temp_ranges
                break
        
        # 2. 데이터 개수에 따른 처리
        # 데이터가 360개 미만이면(예: 1개), 인덱싱(ranges[22])을 할 수 없으므로 LiDAR 포기
        if len(ranges) >= 360:
            image_width = 640
            hfov = 60.0 
            offset_ratio = (image_width / 2 - cx) / image_width 
            target_angle_deg = offset_ratio * hfov
            lidar_idx = int(target_angle_deg) % 360
            
            # 인덱스 접근
            idx_min = (lidar_idx - 1) % 360
            idx_max = (lidar_idx + 1) % 360
            lidar_dist = np.mean([ranges[idx_min], ranges[lidar_idx], ranges[idx_max]])
            
        elif len(ranges) == 1:
            # 센서가 1개뿐이라면, 물체가 '정면'에 있을 때(화면 중앙)만 그 값을 사용
            # 화면 중앙 +- 5픽셀 이내일 때만 신뢰
            if abs(cx - 320) < 5:
                lidar_dist = ranges[0]
            else:
                lidar_dist = 15.0 # 정면이 아니면 센서가 못 봄

        # 3. Vision Fallback
        final_dist = 0.0
        
        # LiDAR 값이 유효한지 체크 (14.5m 미만)
        if lidar_dist < 14.5:
            final_dist = lidar_dist
        else:
            # [Fallback] Vision 기반 거리 추정 (LiDAR 실패 시)
            if bbox_height > 0:
                # 보정값 160.0 (실행해보고 거리가 안 맞으면 숫자 조절)
                focal_length_factor = 160.0 
                final_dist = focal_length_factor / bbox_height
            else:
                final_dist = 15.0

        # 4. 시간 계산
        target_gap = 0.10  # 10cm
        robot_speed = 0.22 
        
        if final_dist <= target_gap:
            time_to_stop = 0.0
        else:
            time_to_stop = (final_dist - target_gap) / robot_speed

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
        print("[TurtlebotFactorySim] Start simulation loop.")
        try:
            while self._running and not self.viewer.should_close():
                # 1) 명령 처리
                self._process_commands()

                # 2) 시뮬레이션 한 스텝
                self.step_simulation()

                # 3) 렌더 + latest_frame 갱신
                self.render()

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
                
                # 접근(APPROACH) 모드: 10cm 도달 시 정지 (TBA)
                elif self.current_action and self.current_action.startswith("APPROACH_"):
                    dist = self.get_dist_to_target(self.search_target_label)

                    if dist is not None:
                        # 10cm(0.1m) 이내면 정지
                        if dist <= 0.10:
                            self.data.ctrl[0] = 0.0
                            self.data.ctrl[1] = 0.0
                            print(f"[TurtlebotFactorySim] Reached 10cm from '{self.search_target_label}' (Dist:{dist:.3f}m) -> Stop.")
                            self.search_target_label = None
                            self.current_action = None
                    else:
                        # 접근 중인데 타겟을 놓친 경우 (예: 너무 가까워서 화면 벗어남 등)
                        # 여기서는 일단 계속 직진하거나 멈추는 정책을 결정해야 함
                        pass
                

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
