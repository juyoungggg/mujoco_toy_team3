import os
import sys
import time
import threading
from queue import Queue

import mujoco as mj
import cv2
import numpy as np

# 프로젝트 루트에서 utils 가져오기
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils.mujoco_renderer import MuJoCoViewer

ACTION_TABLE = {
    "멈춤": (0.0, 0.0),
    "직진": (8.0, 8.0),
    "후진": (-8.0, -8.0),
    "좌회전": (6.0, 8.0),
    "우회전": (8.0, 6.0),
    "제자리 회전": (4.0, -4.0),
}

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


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
        action_end_sim_tim = 0.0,
    ):
        # ===== 경로 설정 =====
        script_path = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(scripts_dir)  # /data/jinsup/js_mujoco

        if xml_path is None:
            xml_path = os.path.join(
                project_root,
                "asset",
                "robotis_tb3",
                "tb3_factory_cards.xml",
            )

        print(f"[TurtlebotFactorySim] Loading scene from: {xml_path}")

        # 검색 모드 타겟 레이블
        self.search_target_label = None  
        self.current_action = current_action
        self.action_end_sim_time = action_end_sim_tim
        # ===== MuJoCo 모델/데이터 로드 =====
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # 기존 MuJoCoViewer 사용
        self.viewer = MuJoCoViewer(self.model, self.data)

        # ===== 카메라 프레임 저장용 =====
        # 항상 "로봇 카메라 기준 BGR 이미지"를 최신 상태로 보관
        self.latest_frame: np.ndarray | None = None

        # ===== YOLO 옵션 =====
        self.use_yolo = use_yolo and _YOLO_AVAILABLE
        self.yolo_conf = yolo_conf
        self.yolo_model = None
        self.yolo_window_name = "Robot YOLO View"
        self._yolo_lock = threading.Lock()

        if self.use_yolo:
            if yolo_weight_path is None:
                raise ValueError("use_yolo=True 인데 yolo_weight_path 가 없습니다.")
            if not os.path.exists(yolo_weight_path):
                raise FileNotFoundError(f"YOLO weight not found: {yolo_weight_path}")

            print(f"[TurtlebotFactorySim] Loading YOLO model: {yolo_weight_path}")
            self.yolo_model = YOLO(yolo_weight_path)
            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.yolo_window_name, 640, 480)
        elif use_yolo and not _YOLO_AVAILABLE:
            print("[TurtlebotFactorySim] ultralytics 가 설치되어 있지 않아 YOLO를 비활성화합니다.")

        # ===== 명령 큐 (LLM / 키보드 등에서 넣어주는 명령) =====
        self.command_queue = command_queue if command_queue is not None else Queue()

        # ===== 루프 설정 =====
        self.fps = fps
        self._running = False

    # ------------------------------------------------------------------
    # 외부에서 사용할 수 있는 유틸 메서드들
    # ------------------------------------------------------------------
    def step_simulation(self):
        """한 타임스텝(fps 기준)만큼 시뮬레이션을 진행."""
        time_prev = self.data.time
        dt = 1.0 / self.fps
        while self.data.time - time_prev < dt:
            self.viewer.step_simulation()

    def render(self):
        """메인뷰 + 로봇 카메라 렌더링, latest_frame 업데이트."""
        # 메인 뷰: IMU overlay
        self.viewer.render_main(overlay_type="imu")

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
            "SEARCH_HEART":   "heart",
            "SEARCH_SPADE":   "spade",   
            "SEARCH_DIAMOND": "diamond",
            "SEARCH_CLUB":    "club",
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
        if not self.use_yolo or self.yolo_model is None:
            return {}
        if self.latest_frame is None:
            return {}

        with self._yolo_lock:
            result = self.yolo_model(self.latest_frame, verbose=False, conf=self.yolo_conf)[0]

        out = {}
        for box in result.boxes:
            cls = int(box.cls[0].item())
            label = result.names.get(cls)
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            out.setdefault(label, []).append({
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2]
            })
        return out
    
    def yolo_detect_image(self):
        if not self.use_yolo or self.yolo_model is None:
            return None
        if self.latest_frame is None:
            return None

        with self._yolo_lock:
            result = self.yolo_model(self.latest_frame, verbose=False, conf=self.yolo_conf)[0]
        return result.plot()

    def _run_yolo_on_latest_frame(self):
        if not self.use_yolo or self.yolo_model is None:
            return
        img_bgr = self.yolo_detect_image()
        if img_bgr is None:
            return
        cv2.imshow(self.yolo_window_name, img_bgr)

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
                if self.search_target_label is not None:
                    det = self.yolo_detect_dict()
                    if self.search_target_label in det:
                        # 타겟 발견 → 정지 + 검색 종료
                        self.data.ctrl[0] = 0.0
                        self.data.ctrl[1] = 0.0
                        print(f"[TurtlebotFactorySim] Found '{self.search_target_label}' → stop search.")
                        self.search_target_label = None
                        self.current_action = None
                        self.action_end_sim_time = 0.0

                # 4) 일반 액션 duration 기반 정지 (검색 모드일 땐 X)
                if (
                    self.current_action 
                    and not (self.current_action.startswith("SEARCH_"))
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
