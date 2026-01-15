import os
from dotenv import load_dotenv
from google import genai

# .env 파일 로드
load_dotenv()

# 환경 변수에서 키 가져오기
api_key = os.environ.get("GEMINI_API_KEY")

# 클라이언트 초기화
client = genai.Client(api_key=api_key)



# 시스템 지침 (System Instructions) 부여
sys_instruction = """
당신은 주어진 텍스트를 바탕으로 튜토리얼을 생성하는 AI 어시스턴트입니다.
텍스트에 어떤 절차를 진행하는 방법에 대한 지침이 포함되어 있다면, 
글머리 기호 목록 형식으로 튜토리얼을 생성하십시오.
그렇지 않으면 텍스트에 지침이 포함되어 있지 않음을 사용자에게 알리십시오.
"""

# 사용자 입력 (지침이 있는 경우)
text_input = """
먼저 평평한 땅을 골라 그라운드 시트를 깝니다. 그 위에 텐트 본체를 펼쳐주세요.
폴대를 조립해서 텐트의 슬리브(구멍)에 X자 모양으로 끼워 넣습니다.
폴대 끝을 텐트 모서리 아일렛(구멍)에 꽂아 텐트를 자립시킵니다.
마지막으로 팩을 45도 각도로 박아서 텐트를 바닥에 고정하고, 
플라이(덮개)를 씌워주면 완성입니다.
하지만 비가 너무 많이 오면 이 텐트는 사용할 수 없습니다.
"""


try:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=genai.types.GenerateContentConfig(
            system_instruction=sys_instruction,
        ),
        contents=text_input,
    )
    print(response.text)
except Exception as e:
    print("키 설정이 잘못되었거나 에러가 발생했습니다:", e)
