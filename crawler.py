import os
import re
import requests
import pandas as pd
from typing import List, Set
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === ① 환경 변수 로드 ===
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("🚨 Tavily API key missing (.env).")
if not OPENAI_API_KEY:
    raise ValueError("🚨 OpenAI API key missing (.env).")

client = OpenAI(api_key=OPENAI_API_KEY)
TAVILY_URL = "https://api.tavily.com/search"

# === ② 검색 쿼리 설정 ===
QUERIES = [
    "AI 교육 스타트업 목록 (예: Riiid, Mathpresso, Sana Labs, Squirrel AI 등)",
    "EdTech 분야에서 인공지능(AI)을 활용하는 주요 스타트업",
    "Top AI education or EdTech startups 2025",
    "AI tutoring platform startups using adaptive learning",
]

# === ③ 정규식 + 필터 설정 ===
NAME_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]*){0,2})\b")

STOPWORDS = {
    "AI", "Labs", "Learning", "Education", "EdTech", "Systems", "Company", "Group",
    "Technology", "Technologies", "Platform", "Startup", "News", "Report", "Top",
    "Software", "Tools", "Adaptive", "Artificial", "Intelligence", "Market",
    "Overview", "Trend", "Global", "Data", "Model", "Classroom", "Program", "School",
}

# === ④ Tavily 검색 함수 ===
def tavily_search(query: str, max_results: int = 30) -> dict:
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "query": query,
        "max_results": min(max_results, 50),
        "include_answer": True,
        "search_depth": "advanced",
    }
    res = requests.post(TAVILY_URL, json=payload, headers=headers, timeout=30)
    if res.status_code != 200:
        print(f"❌ Tavily 검색 실패({res.status_code}) → {res.text}")
        return {}
    return res.json()

# === ⑤ 이름 후보 추출 ===
def extract_candidate_names(*texts: str) -> Set[str]:
    candidates: Set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in NAME_PATTERN.findall(text):
            name = match.strip()
            if len(name) < 3 or name in STOPWORDS:
                continue
            if " " in name and len(name.split()) > 3:
                continue
            if not re.match(r"^[A-Z]", name):
                continue
            candidates.add(name)
    return candidates

# === ⑥ AI 필터 (GPT로 실제 기업명만 남김) ===
def ai_filter_startups(candidates: List[str]) -> List[str]:
    if not candidates:
        return []
    prompt = f"""
다음 리스트 중 실제 'AI 기반 교육(EdTech) 스타트업'이나 기업 이름만 남겨주세요.
뉴스 제목, 기술용어, 일반 단어, 인물, 도시명은 모두 제외하세요.

출력 형식은 회사명만, 한 줄에 하나씩.
리스트:
{candidates}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        return [line.strip() for line in text.splitlines() if line.strip()]
    except Exception as e:
        print("⚠️ AI 필터 실패:", e)
        return candidates

# === ⑦ FAISS 벡터 저장용 함수 ===
def build_faiss_index(all_texts: List[str], output_path: str = "faiss_index"):
    print(f"📦 FAISS 인덱스 생성 중... ({len(all_texts)}개 문서)")
    docs = [Document(page_content=txt) for txt in all_texts if txt.strip()]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(output_path)
    print(f"✅ FAISS 인덱스 저장 완료 → {output_path}/")

# === ⑧ 전체 검색 실행 ===
def aggregate_ai_startups() -> List[str]:
    all_candidates: Set[str] = set()
    all_texts: List[str] = []  # 🔹 RAG용 텍스트 저장
    
    for q in QUERIES:
        print(f"🔍 Searching: {q}")
        data = tavily_search(q)
        if not data:
            continue
        
        # 문서 내용 저장 (RAG에 사용)
        if "answer" in data:
            all_texts.append(data["answer"])
        for r in data.get("results", []):
            all_texts.append(r.get("title", ""))
            all_texts.append(r.get("content", ""))

        # 기업명 추출
        all_candidates.update(extract_candidate_names(data.get("answer", "")))
        for r in data.get("results", []):
            all_candidates.update(extract_candidate_names(r.get("title", ""), r.get("content", "")))

    print(f"🧩 1차 추출된 후보 수: {len(all_candidates)}")
    
    filtered = ai_filter_startups(sorted(all_candidates))
    
    # 🔹 RAG용 FAISS 인덱스 구축
    build_faiss_index(all_texts)
    
    return sorted(set(filtered))

# === ⑨ 실행 ===
if __name__ == "__main__":
    startups = aggregate_ai_startups()
    if startups:
        df = pd.DataFrame(startups, columns=["startup_name"])
        df.to_csv("ai_filtered_startups.csv", index=False, encoding="utf-8-sig")
        print(f"✅ 최종 {len(startups)}개 스타트업 저장 완료 → ai_filtered_startups.csv")
        for s in startups:
            print("-", s)
    else:
        print("❌ 스타트업을 찾지 못했습니다. 쿼리를 변경해보세요.")
