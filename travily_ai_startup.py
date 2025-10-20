import os
import requests
import pandas as pd
import re
from dotenv import load_dotenv

# --- ① 환경 변수 설정 ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("🚨 Tavily API 키가 .env에 설정되어 있지 않습니다!")

# --- ② Tavily 검색 함수 (기사 문서 검색) ---
def search_articles(query, limit=10):
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "max_results": limit,
        "search_depth": "advanced",
        "include_answer": True
    }
    res = requests.post(url, json=payload, headers=headers)
    if res.status_code != 200:
        print(f"❌ 요청 실패({res.status_code}) → {res.text}")
        return []
    data = res.json()
    results = data.get("results", [])
    return results

# --- ③ 스타트업 이름 추출 함수 ---
def extract_startup_names(snippet):
    # 예시 단순정규식: 대문자 시작 단어 + “AI” or “EdTech” 포함 등
    names = re.findall(r"\b([A-Za-z0-9]+(?:\s+A[Ii]| EdTech| Learning| Labs))\b", snippet)
    return list(set(names))

# --- ④ 스타트업 정보 검색 함수 ---
def search_startup_info(name, limit=5):
    query = f"{name} education startup profile funding"
    return search_articles(query, limit=limit)

# --- ⑤ 실행 흐름 ---
if __name__ == "__main__":
    # 1) 기사 검색
    article_query = "AI education startup news 2025 edtech companies using artificial intelligence"
    article_results = search_articles(article_query, limit=20)

    # 2) 기사 결과 → 파일로 저장
    articles_df = pd.DataFrame([
        {
            "rank": i+1,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:300]
        }
        for i, r in enumerate(article_results)
    ])
    articles_df.to_csv("edtech_articles.csv", index=False, encoding="utf-8-sig")
    print("✅ 기사 검색 결과 저장됨 → edtech_articles.csv")

    # 3) 기사들의 snippet에서 스타트업 이름 추출
    startup_names = set()
    for snippet in articles_df["snippet"]:
        for name in extract_startup_names(snippet):
            startup_names.add(name)
    print("🔍 추출된 스타트업 이름들:", startup_names)

    # 4) 스타트업별로 추가 정보 검색
    all_info = []
    for name in startup_names:
        info_results = search_startup_info(name, limit=5)
        for r in info_results:
            all_info.append({
                "startup_name": name,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:300]
            })

    info_df = pd.DataFrame(all_info)
    info_df.to_csv("startup_info.csv", index=False, encoding="utf-8-sig")
    print("✅ 스타트업별 정보 저장됨 → startup_info.csv")
