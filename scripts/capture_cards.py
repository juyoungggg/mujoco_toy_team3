import os
import time
import cv2
import mujoco as mj

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))  # 프로젝트 루트로
print(f"Sys path: {sys.path}")

from utils.mujoco_renderer import MuJoCoViewer

# 사용할 XML 파일 경로
XML_PATH = "/home/juyoung/mujoco_llm/asset/robotis_tb3/tb3_factory_cards.xml"

# 저장 위치 (train 이미지)
OUT_DIR = "img_dataset/images"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)

    viewer = MuJoCoViewer(model, data)

    idx = 0
    frame = 0
    try:
        while not viewer.should_close():
            time_prev = data.time
            while data.time - time_prev < 1.0 / 60.0:
                viewer.step_simulation()

            viewer.render_main()
            viewer.render_robot()
            viewer.poll_events()

            # 프레임 카운트
            frame += 1

            # 10프레임마다 한 번씩만 캡처
            if frame % 10 == 0:
                img = viewer.capture_img()
                out_path = os.path.join(OUT_DIR, f"img_{idx:05d}.jpg")
                cv2.imwrite(out_path, img)
                print("saved:", out_path)
                idx += 1

            time.sleep(0.01)

    finally:
        viewer.terminate()

if __name__ == "__main__":
    main()
