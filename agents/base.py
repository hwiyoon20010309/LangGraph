"""
agents/base.py
공통 State 스키마, 평가 기준, 유틸리티 함수
"""
import os
import re
import requests
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env 파일 로드
load_dotenv(override=True)


# ========================================
# State 스키마 정의
# ========================================

class AgentState(TypedDict, total=False):
    """투자 평가 State - 모든 에이전트가 공유"""
    startup_name: str  # 스타트업
    learning_effectiveness_score: int
    technology_score: int
    growth_potential_score: int
    market_score: int
    competition_score: int
    risk_score: int
    final_judge: Literal["투자", "보류"]
    pdf_path: str  # 보고서 산출 PDF 파일 경로
    # 분석 근거
    learning_effectiveness_analysis_evidence: str
    technology_analysis_evidence: str
    growth_potential_analysis_evidence: str
    market_analysis_evidence: str
    competition_analysis_evidence: str
    report: str


# ========================================
# 평가 기준 정의
# ========================================

EVALUATION_CRITERIA = {
    "technology": [
        "제품이 교육 문제를 명확하게 해결하는가?",
        "AI/ML 기술 활용도가 높은가?",
        "기술의 혁신성과 차별화가 있는가?",
        "기술적 구현 가능성이 높은가?",
        "시스템의 확장 가능성이 있는가?",
        "기술 안정성과 보안이 확보되어 있는가?",
        "데이터 기반 학습 최적화가 가능한가?",
        "API 연동 및 확장성이 뛰어난가?",
        "기술 문서화가 잘 되어 있는가?",
        "오픈소스 활용 및 커뮤니티 기여도가 있는가?"
    ],
    "learning_effectiveness": [
        "학습 성과 측정 지표가 명확한가?",
        "학습자 만족도가 높은가?",
        "학습 완료율이 우수한가?",
        "학습 효과 검증 사례가 있는가?",
        "개인화 학습 지원이 가능한가?",
        "학습 데이터 분석 및 피드백 제공이 되는가?",
        "교사/강사 지원 도구가 있는가?",
        "학습자 참여도 향상 방안이 있는가?",
        "콘텐츠 품질이 우수한가?",
        "학습 경로 추천이 효과적인가?"
    ],
    "market": [
        "타겟 교육 시장 규모가 큰가?",
        "시장 성장률이 높은가?",
        "수익 모델이 명확하고 실현 가능한가?",
        "고객 기반(B2B/B2C)이 확보되어 있는가?",
        "가격 전략이 합리적인가?",
        "시장 진입 전략이 구체적인가?",
        "고객 획득 비용(CAC)이 적절한가?",
        "생애 가치(LTV)가 높은가?",
        "파트너십 확보 가능성이 있는가?",
        "글로벌 시장 진출 가능성이 있는가?"
    ],
    "competition": [
        "경쟁사 대비 명확한 차별화 요소가 있는가?",
        "시장 진입 장벽이 존재하는가?",
        "경쟁 우위(특허, 기술, 네트워크)가 있는가?",
        "브랜드 인지도가 형성되어 있는가?",
        "고객 충성도가 높은가?",
        "선점 효과(First Mover)가 있는가?",
        "네트워크 효과가 작동하는가?",
        "전환 비용(Switching Cost)이 높은가?",
        "경쟁사 대비 가성비가 우수한가?",
        "지속 가능한 경쟁력이 있는가?"
    ],
    "growth_potential": [
        "시장 확장 가능성이 큰가?",
        "제품 다각화 계획이 있는가?",
        "글로벌 진출 전략이 구체적인가?",
        "파트너십 확대 기회가 있는가?",
        "인수합병(M&A) 가능성이 있는가?",
        "IPO 가능성이 있는가?",
        "스케일업을 위한 인프라가 준비되어 있는가?",
        "투자 유치 이력이 있는가?",
        "성장 로드맵이 명확한가?",
        "10배 성장(10x Growth) 가능성이 있는가?"
    ],
    "risk": [
        "재무 리스크가 낮은가?",
        "법적/규제 리스크가 낮은가?",
        "기술 리스크가 낮은가?",
        "시장 리스크가 낮은가?",
        "경영진 리스크가 낮은가?",
        "운영 리스크가 낮은가?",
        "평판 리스크가 낮은가?",
        "경쟁 리스크가 낮은가?",
        "파트너십 의존도 리스크가 낮은가?",
        "확장성 리스크가 낮은가?"
    ]
}


# ========================================
# LLM 초기화
# ========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ========================================
# 유틸리티 함수
# ========================================

def extract_score(analysis: str) -> int:
    """분석 텍스트에서 점수 추출"""
    patterns = [
        r"\*\*총점\*\*[:：]?\s*(\d{1,3})",
        r"총점[:：]?\s*(\d{1,3})\s*(?:점|/100)?",
        r"Score[:：]?\s*(\d{1,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(100, max(0, score))
    
    # 개별 항목 점수 합산
    total = 0
    for i in range(1, 11):
        item_match = re.search(fr"{i}\.\s.*?(\d{{1,2}})\s*(?:/\s*10|점)", analysis, re.DOTALL)
        if item_match:
            total += int(item_match.group(1))
    
    return min(100, max(0, total))


def get_web_context(startup_name: str, query: str) -> str:
    """웹 검색으로 컨텍스트 수집"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    naver_id = os.getenv("NAVER_CLIENT_ID")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    contexts = []
    
    # Tavily API
    if tavily_key:
        try:
            url = "https://api.tavily.com/search"
            params = {"query": f"{startup_name} {query}", "limit": 5}
            headers = {"Authorization": f"Bearer {tavily_key}"}
            res = requests.get(url, params=params, headers=headers, timeout=10)
            items = res.json().get("results", [])
            if items:
                contexts.append("[Tavily 검색]\n" + "\n".join(
                    f"- {it.get('title')} ({it.get('url')})" for it in items
                ))
        except:
            pass
    
    # Naver News API
    if naver_id and naver_secret:
        try:
            url = "https://openapi.naver.com/v1/search/news.json"
            headers = {"X-Naver-Client-Id": naver_id, "X-Naver-Client-Secret": naver_secret}
            params = {"query": f"{startup_name} {query}", "display": 5}
            res = requests.get(url, params=params, headers=headers, timeout=10)
            items = res.json().get("items", [])
            if items:
                contexts.append("[Naver 뉴스]\n" + "\n".join(
                    f"- {it.get('title')} ({it.get('originallink')})" for it in items
                ))
        except:
            pass
    
    return "\n\n".join(contexts) if contexts else "검색 결과 없음"