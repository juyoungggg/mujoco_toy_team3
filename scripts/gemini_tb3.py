import threading
import json
import yaml
import os
import re
from queue import Queue
from dotenv import load_dotenv
from google import genai
from google.genai import types

# YOLO
from ultralytics import YOLO
import cv2


load_dotenv()

TARGET_MAP = {
    "하트": "heart",
    "heart": "heart",
    "다이아": "diamond",
    "다이아몬드": "diamond",
    "diamond": "diamond",
    "클로버": "club",
    "클로바": "club",
    "클럽": "club",
    "club": "club",
    "스페이드": "spade",
    "spade": "spade",
}

SEARCH_CMD = {
    "heart": "SEARCH_HEART",
    "diamond": "SEARCH_DIAMOND",
    "club": "SEARCH_CLUB",
    "spade": "SEARCH_SPADE",
}

# ============================================
# GEMINI LLM RUNNER FOR TURTLEBOT3
# ============================================

class GeminiTb3:
    def __init__(self, prompt_path, model="gemini-robotics-er-1.5-preview", command_queue=None):
        self.command_queue = command_queue if command_queue else Queue()

        # Load prompt.yaml
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_instruction = yaml.safe_load(f)["template"]

        # Gemini client
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model

        # threads
        self.thread = None
        self.stop_event = threading.Event()

    # ----------------------------------------
    def run_gemini(self, question, detection_json):
        """Gemini에게 분석 요청"""
        print(f"[GeminiTb3] Using model: {self.model_name}")
        user_content = f"""
# 감지된 객체 정보(JSON):
{detection_json}

# 질문:
{question}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.1
                ),
                contents=user_content
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {e}"
    # ----------------------------------------
    def _extract_target_from_question(self, q: str) -> str | None:
        q_low = q.lower()
        # 긴 단어 우선(다이아몬드가 다이아보다 먼저 매칭되게)
        keys = sorted(TARGET_MAP.keys(), key=len, reverse=True)
        for k in keys:
            if k.lower() in q_low:
                return TARGET_MAP[k]
        return None

    # ----------------------------------------
    def talk(self, sim):
        while not self.stop_event.is_set():
            try:
                question = input("\n💬 Human: ")

                # YOLO
                det_dict = sim.yolo_detect_dict() or {}
                det_json = json.dumps(det_dict, ensure_ascii=False, indent=2)

                # 목표 카드 추출 (heart/diamond/club/spade)
                target = self._extract_target_from_question(question)

                # 1) 목표가 있는데 화면에 없으면: SEARCH 모드로 전환하고 이 턴은 끝
                if target and target not in det_dict:
                    cmd = SEARCH_CMD[target]
                    print(f"➡️ '{target}' 카드가 안 보여서 {cmd}로 탐색할게요.")
                    self.command_queue.put(cmd)

                    # 여기서 LLM을 호출하면 "멈춤" 같은 액션이 또 들어와서 검색이 끊길 수 있음
                    # 따라서 이 턴은 종료(=검색만 수행)
                    continue

                # 2) 목표가 있거나/없거나 상관없이 LLM 호출 (단, 목표가 있다면 이미 보이는 상태)
                answer = self.run_gemini(question, det_json)
                print(f"\n🤖 Gemini:\n{answer}\n")

                # 3) Action 추출
                action_match = re.search(r"Action:\s*([^\n]+)", answer)
                action = action_match.group(1).strip() if action_match else ""

                # 4) 방어 로직:
                # - 목표가 있고 '보이는' 상태인데 LLM이 멈춤을 내면, 일단 멈춤도 존중하거나
                #   네가 원하면 "target 쪽으로 이동" 같은 룰을 추가할 수도 있음.
                if action:
                    print(f"➡️ Extracted Action: {action}")
                    self.command_queue.put(action)

            except EOFError:
                break
    # ----------------------------------------
    # Gemini + YOLO 스레드 시작
    def start(self, sim):
        self.thread = threading.Thread(target=self.talk, args=(sim,), daemon=True)
        self.thread.start()
        
