import os
import requests
import pandas as pd
from dotenv import load_dotenv
import re

# --- ① 환경 변수 로드 ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("🚨 Tavily API 키가 .env에 설정되어 있지 않습니다!")

# --- ② Tavily AI가 직접 스타트업 이름을 추출하도록 요청 ---
def search_ai_extracted_startups():
    """
    Tavily AI가 전 세계 AI 교육/EdTech 스타트업 이름만 반환하도록 요청
    """
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        # 🔍 한영 병행 쿼리 (AI가 다양한 소스에서 스타트업 이름을 뽑도록)
        "query": (
            "AI 교육(EdTech) 스타트업 목록을 알려줘. "
            "예를 들어 Squirrel AI, Riiid Labs, Sana Labs, GoStudent, BYJU'S 같은 회사처럼 "
            "인공지능(AI)을 활용한 교육 서비스 기업만 포함해줘. "
            "반드시 회사 이름만 리스트 형태로 반환해. "
            "해외 스타트업도 모두 포함해줘. "
            "Return only startup or company names as a clean list (e.g., Squirrel AI, Riiid Labs, Sana Labs, GoStudent, BYJU'S)."
        ),
        "max_results": 50,  # Tavily 허용 최대 검색 수
        "include_answer": True,
        "search_depth": "advanced"
    }

    res = requests.post(url, json=payload, headers=headers)
    if res.status_code != 200:
        print(f"❌ 요청 실패({res.status_code}) → {res.text}")
        return []

    data = res.json()
    answer_text = data.get("answer", "")

    # --- AI가 출력한 텍스트에서 회사명만 정제 ---
    names = re.findall(
        r"\b[A-Z][A-Za-z0-9&\-\s']{2,}(?:AI|Labs|Learning|EdTech|Systems|School|Tech|Education|Academy|Tutors|Inc|Ltd|Company)?\b",
        answer_text
    )

    # 🔧 노이즈 제거 및 중복 제거
    blacklist = {"AI", "Learning", "Education", "School", "Tech", "Labs", "System", "Systems"}
    clean_names = sorted(set(n.strip() for n in names if n.strip() not in blacklist and len(n.strip()) > 2))

    return clean_names

# --- ③ 실행 ---
if __name__ == "__main__":
    startup_names = search_ai_extracted_startups()

    df = pd.DataFrame(startup_names, columns=["startup_name"])
    df.to_csv("ai_extracted_startups.csv", index=False, encoding="utf-8-sig")

    print(f"✅ Tavily AI가 추출한 스타트업 개수: {len(df)}개")
    print("💾 결과 저장 완료 → ai_extracted_startups.csv")
