import os
import sys

# 프로젝트 루트
PROJECT_ROOT = os.path.abspath(os.getcwd())

sys.path.append(PROJECT_ROOT)

# XML 파일 경로 설정
xml_path = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory.xml")
print("Using XML:", xml_path)

from scripts.tb3_sim import TurtlebotFactorySim

if __name__ == "__main__":

    # simulator 실행 (YOLO 탐지 기능은 끄고 simulation만 확인)
    sim = TurtlebotFactorySim(xml_path=xml_path, use_yolo=False)
    sim.start()
