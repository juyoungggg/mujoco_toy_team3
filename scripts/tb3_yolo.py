import os
import sys

# 프로젝트 루트 추가 (js_mujoco/scripts 기준 ipynb라고 가정)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# 통합 시뮬 클래스 import
from tb3_sim import TurtlebotFactorySim

# XML, YOLO weight 경로
PROJECT_ROOT = os.path.abspath(os.getcwd())
XML_PATH = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory_cards.xml")
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "scripts", "best.pt")  # 위치에 맞게 수정

print("XML:", XML_PATH)
print("YOLO:", YOLO_WEIGHTS)

# 터틀봇 + YOLO 통합 시뮬 실행

# tb3_sim:
# - MuJoCo 로드
# - MuJoCoViewer 생성
# - latest_frame 업데이트
# - YOLO 로드 & OpenCV 창 띄우기까지 다 처리
if __name__ == "__main__":

    sim = TurtlebotFactorySim(
        xml_path=XML_PATH,
        use_yolo=True,              # YOLO 함께 사용
        yolo_weight_path=YOLO_WEIGHTS,
        yolo_conf=0.5,
    )

    sim.start()   # 내부에서 while 루프 + 렌더링 + YOLO + cv2.imshow("Robot YOLO View") 수행
