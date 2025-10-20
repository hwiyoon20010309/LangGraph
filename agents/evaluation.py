import os
import re
import requests
from datetime import datetime
from typing import TypedDict, Literal, List, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# .env 파일 로드
load_dotenv()


# ========================================
# 1. State 스키마 정의
# ========================================

class InvestmentState(TypedDict):
    """투자 평가 State - 모든 에이전트가 공유"""
    
    # 기본 정보
    startup_name: str
    
    # 각 에이전트 분석 결과
    technology_score: int
    technology_evidence: str
    
    learning_effectiveness_score: int
    learning_effectiveness_evidence: str
    
    market_score: int
    market_evidence: str
    
    competition_score: int
    competition_evidence: str
    
    growth_potential_score: int
    growth_potential_evidence: str
    
    # 종합 판단 결과
    total_score: int
    investment_decision: Literal["투자", "보류"]
    decision_reasoning: str
    
    # 최종 보고서
    final_report: str
    report_path: str


# ========================================
# 2. 평가 기준 정의
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
    ]
}

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ========================================
# 3. 유틸리티 함수
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


# ========================================
# 4. 독립적인 분석 Agent들
# ========================================

def technology_agent(state: InvestmentState) -> InvestmentState:
    """Agent 1: 기술력 분석"""
    print("\n🔧 [Agent 1] 기술력 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["technology"]
    context = get_web_context(startup_name, "교육 기술 혁신")
    
    prompt = ChatPromptTemplate.from_template("""
교육 스타트업 '{startup_name}'의 기술력을 평가하세요.

**평가 기준 (각 항목 0-10점):**
{checklist}

**참고 자료:**
{context}

**출력 형식:**
각 항목별로:
- 점수 (0-10점)
- 근거 (URL 포함)

마지막에 **총점: [0-100 숫자]** 형식으로 작성하세요.
""")
    
    response = (prompt | llm).invoke({
        "startup_name": startup_name,
        "checklist": "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist)),
        "context": context
    })
    
    analysis = response.content
    score = extract_score(analysis)
    
    state["technology_score"] = score
    state["technology_evidence"] = analysis
    
    print(f"✅ [Agent 1] 완료 - 기술력 점수: {score}")
    return state


def learning_effectiveness_agent(state: InvestmentState) -> InvestmentState:
    """Agent 2: 학습 효과성 분석"""
    print("\n📚 [Agent 2] 학습 효과성 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["learning_effectiveness"]
    context = get_web_context(startup_name, "학습 효과 성과")
    
    prompt = ChatPromptTemplate.from_template("""
교육 스타트업 '{startup_name}'의 학습 효과성을 평가하세요.

**평가 기준 (각 항목 0-10점):**
{checklist}

**참고 자료:**
{context}

**출력 형식:**
각 항목별로:
- 점수 (0-10점)
- 근거 (URL 포함)

마지막에 **총점: [0-100 숫자]** 형식으로 작성하세요.
""")
    
    response = (prompt | llm).invoke({
        "startup_name": startup_name,
        "checklist": "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist)),
        "context": context
    })
    
    analysis = response.content
    score = extract_score(analysis)
    
    state["learning_effectiveness_score"] = score
    state["learning_effectiveness_evidence"] = analysis
    
    print(f"✅ [Agent 2] 완료 - 학습효과 점수: {score}")
    return state


def market_agent(state: InvestmentState) -> InvestmentState:
    """Agent 3: 시장성 분석 (RAG 포함)"""
    print("\n💰 [Agent 3] 시장성 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["market"]
    
    # PDF RAG
    pdf_path = os.getenv("PDF_PATH", "")
    rag_context = ""
    if pdf_path and os.path.exists(pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
            vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved = retriever.get_relevant_documents(f"{startup_name} 교육 시장")
            rag_context = "\n".join([doc.page_content for doc in retrieved])
        except:
            rag_context = "PDF 로딩 실패"
    
    # Web Search
    try:
        search = TavilySearchResults(k=10)
        results = search.invoke(f"{startup_name} 교육 시장 규모")
        web_context = "\n".join([f"- {r['title']} ({r['url']})" for r in results])
    except:
        web_context = "검색 실패"
    
    combined = f"[PDF 자료]\n{rag_context}\n\n[웹 검색]\n{web_context}"
    
    prompt = ChatPromptTemplate.from_template("""
교육 스타트업 '{startup_name}'의 시장성을 평가하세요.

**평가 기준 (각 항목 0-10점):**
{checklist}

**참고 자료:**
{context}

**출력 형식:**
각 항목별로:
- 점수 (0-10점)
- 근거 (URL 포함)

마지막에 **총점: [0-100 숫자]** 형식으로 작성하세요.
""")
    
    response = (prompt | llm).invoke({
        "startup_name": startup_name,
        "checklist": "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist)),
        "context": combined
    })
    
    analysis = response.content
    score = extract_score(analysis)
    
    state["market_score"] = score
    state["market_evidence"] = analysis
    
    print(f"✅ [Agent 3] 완료 - 시장성 점수: {score}")
    return state


def competition_agent(state: InvestmentState) -> InvestmentState:
    """Agent 4: 경쟁력 분석"""
    print("\n⚔️ [Agent 4] 경쟁력 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["competition"]
    
    try:
        search = TavilySearchResults(k=15)
        results = search.invoke(f"{startup_name} 경쟁사 비교")
        context = "\n".join([f"- {r['title']} ({r['url']})" for r in results])
    except:
        context = "검색 실패"
    
    prompt = ChatPromptTemplate.from_template("""
교육 스타트업 '{startup_name}'의 경쟁력을 평가하세요.

**평가 기준 (각 항목 0-10점):**
{checklist}

**참고 자료:**
{context}

**출력 형식:**
각 항목별로:
- 점수 (0-10점)
- 근거 (URL 포함)

마지막에 **총점: [0-100 숫자]** 형식으로 작성하세요.
""")
    
    response = (prompt | llm).invoke({
        "startup_name": startup_name,
        "checklist": "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist)),
        "context": context
    })
    
    analysis = response.content
    score = extract_score(analysis)
    
    state["competition_score"] = score
    state["competition_evidence"] = analysis
    
    print(f"✅ [Agent 4] 완료 - 경쟁력 점수: {score}")
    return state


def growth_potential_agent(state: InvestmentState) -> InvestmentState:
    """Agent 5: 성장 가능성 분석"""
    print("\n🚀 [Agent 5] 성장 가능성 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["growth_potential"]
    context = get_web_context(startup_name, "성장 가능성 투자 유치")
    
    prompt = ChatPromptTemplate.from_template("""
교육 스타트업 '{startup_name}'의 성장 가능성을 평가하세요.

**평가 기준 (각 항목 0-10점):**
{checklist}

**참고 자료:**
{context}

**출력 형식:**
각 항목별로:
- 점수 (0-10점)
- 근거 (URL 포함)

마지막에 **총점: [0-100 숫자]** 형식으로 작성하세요.
""")
    
    response = (prompt | llm).invoke({
        "startup_name": startup_name,
        "checklist": "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist)),
        "context": context
    })
    
    analysis = response.content
    score = extract_score(analysis)
    
    state["growth_potential_score"] = score
    state["growth_potential_evidence"] = analysis
    
    print(f"✅ [Agent 5] 완료 - 성장가능성 점수: {score}")
    return state


# ========================================
# 5. 종합 판단 Agent
# ========================================

def comprehensive_judge_agent(state: InvestmentState) -> InvestmentState:
    """Agent 6: 종합 판단 - State의 모든 점수를 기반으로 투자 결정"""
    print("\n⚖️ [Agent 6] 종합 판단 시작...")
    
    # State에서 점수 수집
    tech = state["technology_score"]
    learning = state["learning_effectiveness_score"]
    market = state["market_score"]
    competition = state["competition_score"]
    growth = state["growth_potential_score"]
    
    # 가중 평균 계산
    weights = {
        "tech": 0.25,
        "learning": 0.20,
        "market": 0.25,
        "competition": 0.15,
        "growth": 0.15
    }
    
    total_score = int(
        tech * weights["tech"] +
        learning * weights["learning"] +
        market * weights["market"] +
        competition * weights["competition"] +
        growth * weights["growth"]
    )
    
    prompt = ChatPromptTemplate.from_template("""
다음 점수를 바탕으로 투자 결정을 내리세요:

**점수 현황:**
- 기술력: {tech}/100 (가중치 25%)
- 학습효과: {learning}/100 (가중치 20%)
- 시장성: {market}/100 (가중치 25%)
- 경쟁력: {competition}/100 (가중치 15%)
- 성장가능성: {growth}/100 (가중치 15%)

**가중 평균 총점: {total}/100**

**판단 기준:**
- 총점 70 이상 AND 모든 항목 50 이상 → "투자"
- 총점 50-69 OR 일부 항목 50 미만 → "보류"
- 총점 50 미만 → "보류"

**출력 형식:**
1. 결정: [투자/보류]
2. 근거: (각 항목별 강점/약점 분석)
3. 개선 제안: (보류인 경우)
""")
    
    response = (prompt | llm).invoke({
        "tech": tech,
        "learning": learning,
        "market": market,
        "competition": competition,
        "growth": growth,
        "total": total_score
    })
    
    reasoning = response.content
    
    # 투자 결정 추출
    decision = "보류"
    if total_score >= 70 and all(s >= 50 for s in [tech, learning, market, competition, growth]):
        decision = "투자"
    
    state["total_score"] = total_score
    state["investment_decision"] = decision
    state["decision_reasoning"] = reasoning
    
    print(f"✅ [Agent 6] 완료 - 최종 결정: {decision} (총점: {total_score})")
    return state


# ========================================
# 6. 보고서 생성 Agent
# ========================================

def report_generation_agent(state: InvestmentState) -> InvestmentState:
    """Agent 7: 최종 보고서 생성 - State 기반"""
    print("\n📝 [Agent 7] 보고서 생성 시작...")
    
    prompt = ChatPromptTemplate.from_template("""
# 투자 심사 보고서

## 기본 정보
- 스타트업: {startup_name}
- 작성일: {date}
- 최종 결정: **{decision}**
- 총점: **{total}/100**

## 항목별 점수
| 항목 | 점수 | 가중치 |
|------|------|--------|
| 기술력 | {tech}/100 | 25% |
| 학습효과 | {learning}/100 | 20% |
| 시장성 | {market}/100 | 25% |
| 경쟁력 | {competition}/100 | 15% |
| 성장가능성 | {growth}/100 | 15% |

## 종합 판단
{reasoning}

## 상세 분석

### 1. 기술력 분석
{tech_evidence}

### 2. 학습 효과성 분석
{learning_evidence}

### 3. 시장성 분석
{market_evidence}

### 4. 경쟁력 분석
{competition_evidence}

### 5. 성장 가능성 분석
{growth_evidence}

## 결론
위 분석 결과를 바탕으로 작성된 전문적인 투자 심사 보고서를 최종 정리해주세요.
강점, 약점, 기회, 위협 요인을 SWOT 형태로 정리하고,
투자 결정에 대한 명확한 권고사항을 제시하세요.
""")
    
    response = (prompt | llm).invoke({
        "startup_name": state["startup_name"],
        "date": datetime.now().strftime("%Y년 %m월 %d일"),
        "decision": state["investment_decision"],
        "total": state["total_score"],
        "tech": state["technology_score"],
        "learning": state["learning_effectiveness_score"],
        "market": state["market_score"],
        "competition": state["competition_score"],
        "growth": state["growth_potential_score"],
        "reasoning": state["decision_reasoning"],
        "tech_evidence": state["technology_evidence"],
        "learning_evidence": state["learning_effectiveness_evidence"],
        "market_evidence": state["market_evidence"],
        "competition_evidence": state["competition_evidence"],
        "growth_evidence": state["growth_potential_evidence"]
    })
    
    state["final_report"] = response.content
    
    # 파일 저장
    output_dir = "investment_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{state['startup_name']}_투자분석_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(state["final_report"])
    
    state["report_path"] = filepath
    
    print(f"✅ [Agent 7] 완료 - 보고서 저장: {filepath}")
    return state


# ========================================
# 7. LangGraph 워크플로우 구성
# ========================================

def build_agent_workflow():
    """독립적인 Agent 기반 워크플로우 구축"""
    
    workflow = StateGraph(InvestmentState)
    
    # Agent 노드 추가
    workflow.add_node("technology", technology_agent)
    workflow.add_node("learning", learning_effectiveness_agent)
    workflow.add_node("market", market_agent)
    workflow.add_node("competition", competition_agent)
    workflow.add_node("growth", growth_potential_agent)
    workflow.add_node("judge", comprehensive_judge_agent)
    workflow.add_node("report", report_generation_agent)
    
    # 순차 실행 플로우
    workflow.set_entry_point("technology")
    workflow.add_edge("technology", "learning")
    workflow.add_edge("learning", "market")
    workflow.add_edge("market", "competition")
    workflow.add_edge("competition", "growth")
    workflow.add_edge("growth", "judge")
    workflow.add_edge("judge", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# ========================================
# 8. 실행 함수
# ========================================

def run_investment_analysis(startup_name: str):
    """투자 분석 실행"""
    
    print("=" * 70)
    print(f"🎯 투자 심사 시작: {startup_name}")
    print("=" * 70)
    
    # 초기 State
    initial_state: InvestmentState = {
        "startup_name": startup_name,
        "technology_score": 0,
        "technology_evidence": "",
        "learning_effectiveness_score": 0,
        "learning_effectiveness_evidence": "",
        "market_score": 0,
        "market_evidence": "",
        "competition_score": 0,
        "competition_evidence": "",
        "growth_potential_score": 0,
        "growth_potential_evidence": "",
        "total_score": 0,
        "investment_decision": "미결정",
        "decision_reasoning": "",
        "final_report": "",
        "report_path": ""
    }
    
    # Agent 워크플로우 실행
    agent = build_agent_workflow()
    final_state = agent.invoke(initial_state)
    
    # 최종 결과 출력
    print("\n" + "=" * 70)
    print("📊 최종 결과")
    print("=" * 70)
    print(f"스타트업: {final_state['startup_name']}")
    print(f"투자 결정: {final_state['investment_decision']}")
    print(f"총점: {final_state['total_score']}/100")
    print(f"\n항목별 점수:")
    print(f"  🔧 기술력: {final_state['technology_score']}")
    print(f"  📚 학습효과: {final_state['learning_effectiveness_score']}")
    print(f"  💰 시장성: {final_state['market_score']}")
    print(f"  ⚔️ 경쟁력: {final_state['competition_score']}")
    print(f"  🚀 성장가능성: {final_state['growth_potential_score']}")
    print(f"\n📄 보고서: {final_state['report_path']}")
    print("=" * 70)
    
    return final_state


# ========================================
# 9. 메인 실행
# ========================================

if __name__ == "__main__":
    startup = input("분석할 교육 스타트업 이름: ").strip()
    
    if not startup:
        print("❌ 스타트업 이름을 입력해주세요.")
    else:
        result = run_investment_analysis(startup)
        
        # 보고서 미리보기
        print("\n" + "=" * 70)
        print("📄 보고서 미리보기")
        print("=" * 70)
        preview = result["final_report"][:1000]
        print(preview + "..." if len(result["final_report"]) > 1000 else preview)
        print("=" * 70)