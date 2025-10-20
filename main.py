# -*- coding: utf-8 -*-
"""
EdTech 투자 파이프라인 (Flowchart 기반 / Tavily 통합 RAG)
- Flow:
  Start -> WebCrawling -> Filtering -> SelectOne
         -> TechSummary -> MarketEval -> CompAnalysis -> InvestDecision -> (Hold -> SelectOne | Report -> End)

기능:
- [수정] Tavily 검색으로 '한국 교육 스타트업' 리스트 및 뉴스 수집 → FAISS RAG 인덱스
- 스타트업 후보 자동 추출(LLM) + 간단 선정 점수(키워드 기반)
- 6개 평가영역(기술력, 학습성과, 시장성, 경쟁력, 리스크, 성장가능성) LLM 점수화
- 최종 투자판단 + PDF 보고서 생성

필요:
- pip install langgraph langchain langchain-openai langchain-community faiss-cpu tiktoken requests beautifulsoup4 reportlab python-dotenv rich
- .env:
    OPENAI_API_KEY=sk-...
    TAVILY_API_KEY=tvly-...   (필수: Tavily API 키가 없으면 이 코드는 동작하지 않습니다)
"""

import os, re, json, requests
import time
import traceback
from typing import TypedDict, Literal, List, Dict, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich import print

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
# [💡💡💡] SqliteSaver import 구문 삭제됨
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# 환경 변수 로드
# =========================
from dotenv import load_dotenv
import os

# .env 파일 불러오기
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# 기본 체크
if not OPENAI_API_KEY:
    raise RuntimeError("❌ 환경변수 OPENAI_API_KEY가 필요합니다. (.env 파일 또는 시스템 환경변수 확인)")
else:
    print("✅ OPENAI_API_KEY 로드 완료")

# [수정] Tavily는 이제 선택이 아닌 필수입니다.
if not TAVILY_API_KEY:
    raise RuntimeError("❌ 환경변수 TAVILY_API_KEY가 필요합니다. 이 파이프라인은 Tavily 검색에 의존합니다.")
else:
    print("✅ TAVILY_API_KEY 로드 완료")

# LangChain Tracing 설정 확인용 출력 (선택)
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print(f"🧠 LangChain tracing 활성화됨 — 프로젝트: {LANGCHAIN_PROJECT}")


# =========================
# 상태 스키마 (공유 State)
# =========================
class AgentState(TypedDict, total=False):
    # 후보/선정
    candidates: List[Dict]
    filtered: List[Dict]
    current_candidate: Dict
    current_idx: int
    loop_count: int

    # 6개 점수 (0~100)
    technology_score: int
    learning_effectiveness_score: int
    market_score: int
    competition_score: int
    risk_score: int
    growth_potential_score: int

    # 분석 근거
    technology_analysis_evidence: str
    learning_effectiveness_analysis_evidence: str
    market_analysis_evidence: str
    competition_analysis_evidence: str
    risk_analysis_evidence: str
    growth_potential_analysis_evidence: str

    # 판단/산출
    final_judge: Literal["투자", "보류"]
    report: str
    pdf_path: str

    # RAG 인덱스 경로
    vectorstore_path: str


# =========================
# 평가 기준(참조용) / 임계치
# =========================
EVALUATION_CRITERIA = {
    "technology": {
        "제품이 교육 문제를 명확하게 해결하는가?": 10,
        "AI/ML 기술 활용도가 높은가?": 10,
        "기술의 혁신성과 차별화가 있는가?": 10,
        "기술적 구현 가능성이 높은가?": 10,
        "시스템의 확장 가능성이 있는가?": 10,
        "기술 안정성과 보안이 확보되어 있는가?": 10,
        "데이터 기반 학습 최적화가 가능한가?": 10,
        "API 연동 및 확장성이 뛰어난가?": 10,
        "기술 문서화가 잘 되어 있는가?": 10,
        "오픈소스 활용 및 커뮤니티 기여도가 있는가?": 10
    },
    "learning_effectiveness": {
        "학습 성과 측정 지표가 명확한가?": 10,
        "학습자 만족도가 높은가?": 10,
        "학습 완료율이 우수한가?": 10,
        "학습 효과 검증 사례가 있는가?": 10,
        "개인화 학습 지원이 가능한가?": 10,
        "학습 데이터 분석 및 피드백 제공이 되는가?": 10,
        "교사/강사 지원 도구가 있는가?": 10,
        "학습자 참여도 향상 방안이 있는가?": 10,
        "콘텐츠 품질이 우수한가?": 10,
        "학습 경로 추천이 효과적인가?": 10
    },
    "market": {
        "타겟 교육 시장 규모가 큰가?": 10,
        "시장 성장률이 높은가?": 10,
        "수익 모델이 명확하고 실현 가능한가?": 10,
        "고객 기반(B2B/B2C)이 확보되어 있는가?": 10,
        "가격 전략이 합리적인가?": 10,
        "시장 진입 전략이 구체적인가?": 10,
        "CAC이 적절한가?": 10,
        "LTV가 높은가?": 10,
        "파트너십 확보 가능성이 있는가?": 10,
        "글로벌 시장 진출 가능성이 있는가?": 10
    },
    "competition": {
        "경쟁사 대비 명확한 차별화 요소가 있는가?": 10,
        "시장 진입 장벽이 존재하는가?": 10,
        "경쟁 우위(특허, 기술, 네트워크)가 있는가?": 10,
        "브랜드 인지도가 형성되어 있는가?": 10,
        "고객 충성도가 높은가?": 10,
        "선점 효과(First Mover)가 있는가?": 10,
        "네트워크 효과가 작동하는가?": 10,
        "전환 비용(Switching Cost)이 높은가?": 10,
        "경쟁사 대비 가성비가 우수한가?": 10,
        "지속 가능한 경쟁력이 있는가?": 10
    },
    "risk": {
        "재무 안정성이 확보되어 있는가?": 10,
        "창업자 및 팀 역량이 우수한가?": 10,
        "교육 규제 리스크가 낮은가?": 10,
        "사업 지속 가능성이 있는가?": 10,
        "기술 리스크 대응이 되는가?": 10,
        "법률 리스크(개인정보 등)가 낮은가?": 10,
        "운영 리스크가 관리되는가?": 10,
        "시장 변화 대응력이 있는가?": 10,
        "의존성 리스크(특정 고객/파트너)가 낮은가?": 10,
        "위기 관리 체계가 갖춰져 있는가?": 10
    },
    "growth_potential": {
        "시장 확장 가능성이 큰가?": 10,
        "제품 다각화 계획이 있는가?": 10,
        "글로벌 진출 전략이 구체적인가?": 10,
        "파트너십 확대 기회가 있는가?": 10,
        "M&A 가능성이 있는가?": 10,
        "IPO 가능성이 있는가?": 10,
        "스케일업 인프라가 준비되어 있는가?": 10,
        "투자 유치 이력이 있는가?": 10,
        "성장 로드맵이 명확한가?": 10,
        "10배 성장(10x) 가능성이 있는가?": 10
    }
}
INVESTMENT_THRESHOLDS = {"투자": 70, "보류": 0}

# =========================
# 공통 유틸
# =========================
def clean_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

def tavily_search_docs(query: str, limit: int = 8) -> List[Document]:
    """Tavily API로 검색 결과를 문서 리스트로 변환"""
    if not TAVILY_API_KEY:
        return []
    try:
        url = "https://api.tavily.com/v1/search"
        params = {"query": query, "max_results": limit, "include_raw_content": True} # 원본 콘텐츠 포함
        headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
        res = requests.get(url, params=params, headers=headers, timeout=20)
        res.raise_for_status()
        data = res.json()
        items = data.get("results") or data.get("items") or []
        docs: List[Document] = []
        for it in items:
            title = it.get("title") or it.get("url") or "untitled"
            url_ = it.get("url")
            content = it.get("raw_content") or it.get("content") or it.get("snippet") or ""
            if content:
                docs.append(Document(page_content=clean_text(content)[:15000],
                                     metadata={"url": url_, "title": title}))
        return docs
    except Exception as e:
        print(f"❌ Tavily 검색 중 오류 발생: {e}")
        return []

def build_vectorstore(docs: List[Document], save_path: str):
    os.makedirs(save_path, exist_ok=True)
    vs = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    vs.save_local(save_path)
    return save_path

def load_vectorstore(path: str):
    return FAISS.load_local(path, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)


# ====================================================
# 스타트업 후보 추출(RAG) & 선정 점수화(키워드 기반)
# ====================================================
CAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 EdTech VC 스카우터입니다. 문서 스니펫을 기반으로 '교육(EdTech) 분야 스타트업' 후보를 최대 10개 JSON 배열로 뽑으세요. "
     "각 항목은 name, url, region, stage, last_funding, blurb(한줄설명) 필드를 포함해야 합니다. 오직 JSON만 출력."),
    ("human", "문서 스니펫:\n{snippets}")
])

def extract_candidates_with_rag(vs, query="South Korea EdTech startups list 2024 2025") -> List[Dict]:
    retriever = vs.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        print("⚠️ RAG 후보 추출: 관련 문서를 찾지 못함")
        return []
    text = "\n\n".join(
        f"- {d.metadata.get('title','')} | {d.metadata.get('url','')}\n{d.page_content[:1500]}"
        for d in docs
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=OPENAI_API_KEY)
    raw = llm.invoke(CAND_PROMPT.format_messages(snippets=text)).content
    try:
        data = json.loads(raw)
    except Exception:
        try:
            raw = raw[raw.find("["): raw.rfind("]")+1]
            data = json.loads(raw)
        except Exception:
            data = []
    # normalize
    out = []
    for x in data:
        out.append({
            "name": (x.get("name") or "").strip(),
            "url": x.get("url"),
            "region": x.get("region") or "South Korea",
            "stage": x.get("stage"),
            "last_funding": x.get("last_funding"), # [💡] 잘렸던 부분 복구
            "blurb": x.get("blurb")
        })
    return [c for c in out if c["name"]]

def keyword_score(text: str, keywords: List[str], unit: float = 0.2) -> float:
    count = sum(1 for kw in keywords if re.search(kw, text, re.I))
    return min(count * unit, 1.0)

def selection_score(c: Dict) -> float:
    desc = " ".join(filter(None, [c.get("name",""), c.get("stage",""), c.get("last_funding",""), c.get("region",""), c.get("blurb","")]))
    score = 0.0
    score += 0.25 * keyword_score(desc, ["growth","10x","rapid","scale up","fast","expansion"])
    score += 0.25 * keyword_score(desc, ["AI","LLM","innovative","unique","creative","patent"])
    score += 0.20 * keyword_score(desc, ["global","scalable","leader","first mover","market share","expansion"])
    score -= 0.15 * keyword_score(desc, ["risk","uncertain","volatile","unstable"])
    score += 0.10 * keyword_score(desc, ["VC","venture","angel","series","funding"])
    score += 0.05 * keyword_score(desc, ["IPO","M&A","exit","acquisition"])
    return max(0.0, min(round(score, 3), 1.0))


# =========================
# 분석 프롬프트(6개 영역)
# =========================
PROMPT_ANALYZE = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 투자심사역입니다. 회사와 문서 스니펫을 근거로 {category}을(를) 0~100점으로 평가하고, "
     "한 문장 근거를 제시하세요. JSON만 출력. 예:{\"score\":82,\"evidence\":\"...\"}"),
    ("human", "회사: {company}\n문서 스니펫:\n{snips}")
])

def rag_score(vs, company: str, category_desc: str, search_q: str) -> Dict:
    retriever = vs.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(f"{company} {search_q}")
    snips = "\n\n".join(
        f"- {d.metadata.get('title','')} | {d.metadata.get('url','')}\n{d.page_content[:1500]}"
        for d in docs
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    raw = llm.invoke(PROMPT_ANALYZE.format_messages(company=company, snips=snips, category=category_desc)).content
    try:
        data = json.loads(raw)
    except Exception:
        data = {"score": 60, "evidence": "출처 부족"}
    score = int(max(0, min(100, data.get("score", 60))))
    ev = str(data.get("evidence",""))
    return {"score": score, "evidence": ev}


# =========================
# Node 구현 (Flowchart 매핑)
# =========================
def node_web_crawling(state: AgentState) -> Dict: 
    print("🌐 WebCrawling: Tavily로 스타트업 정보 수집/RAG 인덱스 구축 중...")
    search_query = "South Korea EdTech startups list 2024 2025"
    
    try:
        docs: List[Document] = []
        if TAVILY_API_KEY:
            print(f"✅ Tavily로 '{search_query}' 정보 검색 중...")
            docs = tavily_search_docs(search_query, limit=15) 
            if not docs:
                 print("⚠️ Tavily 검색 결과가 없습니다.")
            else:
                 print(f"✅ Tavily에서 {len(docs)}개 문서 수집 완료")
        else:
            print("❌ TAVILY_API_KEY가 없어 크롤링할 소스가 없습니다.")
            return {"vectorstore_path": "error", "candidates": []}

        if not docs:
            print("⚠️ 수집된 문서가 없어 RAG 인덱스를 구축할 수 없습니다.")
            return {"vectorstore_path": "error", "candidates": []}

        vs_path = "vs_edtech_index"
        os.makedirs(vs_path, exist_ok=True)
        vs_path_result = build_vectorstore(docs, vs_path) 
        vs = load_vectorstore(vs_path_result)

        cands = extract_candidates_with_rag(vs, query=search_query) 
        if cands:
            print(f"  ↳ 후보 {len(cands)}개 추출")
        else:
            print("⚠️ 후보 없음 — 다음 단계에서 자동 보류 처리")
            
        return {"vectorstore_path": vs_path_result, "candidates": cands}
        
    except Exception as e:
        print(f"❌ WebCrawling 중 치명적 오류 발생: {e}")
        return {"vectorstore_path": "error", "candidates": []}


def node_filtering(state: AgentState) -> Dict: 
    print("[bold cyan]🔍 Filtering: 기준 부합 후보 선별 중...[/]")
    cands = state.get("candidates", [])
    filtered = [c for c in cands if c.get("name") and (c.get("blurb") or c.get("url"))]
    loop_count = state.get("loop_count", 0) + 1
    
    ranked = sorted(filtered, key=selection_score, reverse=True)
    print(f"  ↳ 필터 통과 (정렬 완료): [bold]{len(ranked)}개[/]")
    
    return {"filtered": ranked, "current_idx": 0, "loop_count": loop_count}


def node_select_one(state: AgentState) -> Dict: 
    filtered = state.get("filtered", [])
    idx = state.get("current_idx", 0)
    
    current_candidate = filtered[idx] if idx < len(filtered) else None
    
    if current_candidate:
        n = current_candidate["name"]
        sc = selection_score(current_candidate)
        print(f"[bold green]✅ SelectOne:[/] #{idx+1} {n} (선정점수 {sc})")
    else:
        print("[bold yellow]⚠️ SelectOne: 더 이상 평가할 후보 없음[/]")
        
    return {"current_candidate": current_candidate, "current_idx": idx + 1}


def node_tech_summary(state: AgentState) -> Dict: 
    c = state.get("current_candidate")
    if not c:
        return {"technology_score": 0, "learning_effectiveness_score": 0} 

    try:
        company = c["name"]
        vs = load_vectorstore(state["vectorstore_path"])
        tech = rag_score(vs, company, "기술력(AI/ML 활용, 혁신성, 확장성)", "architecture model patent scalability AI LLM")
        learn = rag_score(vs, company, "학습 성과(학습효과, 만족도, 완료율)", "learning outcome efficacy satisfaction completion rate case study")
        print(f"[bold cyan]🧠 TechSummary:[/] 기술 {tech['score']}, 학습성과 {learn['score']}")
        return {
            "technology_score": tech["score"],
            "technology_analysis_evidence": tech["evidence"],
            "learning_effectiveness_score": learn["score"],
            "learning_effectiveness_analysis_evidence": learn["evidence"]
        }
    except Exception as e:
        print(f"❌ TechSummary 중 오류: {e}")
        return { 
            "technology_score": 0, "technology_analysis_evidence": f"분석 오류: {e}",
            "learning_effectiveness_score": 0, "learning_effectiveness_analysis_evidence": f"분석 오류: {e}"
        }


def node_market_eval(state: AgentState) -> Dict: 
    c = state.get("current_candidate")
    if not c:
        return {"market_score": 0, "growth_potential_score": 0}

    try:
        company = c["name"]
        vs = load_vectorstore(state["vectorstore_path"])
        market = rag_score(vs, company, "시장성(시장규모/성장률/수익모델)", "market size growth TAM SAM SOM revenue model pricing")
        growth = rag_score(vs, company, "성장 가능성(시장 확장/글로벌 진출)", "expansion global go-to-market partnership localization")
        print(f"[bold cyan]📈 MarketEval:[/] 시장 {market['score']}, 성장가능성 {growth['score']}")
        
        return {
            "market_score": market["score"],
            "market_analysis_evidence": market["evidence"],
            "growth_potential_score": growth["score"],
            "growth_potential_analysis_evidence": growth["evidence"]
        }
    except Exception as e:
        print(f"❌ MarketEval 중 오류: {e}")
        return {
            "market_score": 0, "market_analysis_evidence": f"분석 오류: {e}",
            "growth_potential_score": 0, "growth_potential_analysis_evidence": f"분석 오류: {e}"
        }


def node_comp_analysis(state: AgentState) -> Dict: 
    c = state.get("current_candidate")
    if not c:
        return {"competition_score": 0, "risk_score": 0} 

    try:
        company = c["name"]
        vs = load_vectorstore(state["vectorstore_path"])
        comp = rag_score(vs, company, "경쟁력(차별화/경쟁우위/모방난이도)", "competitor differentiation moat switching cost")
        risk = rag_score(vs, company, "리스크(재무/팀/규제)", "risk runway funding debt regulation compliance team risk")
        print(f"[bold cyan]⚔️ CompAnalysis:[/] 경쟁력 {comp['score']}, 리스크 {risk['score']}")
        
        return {
            "competition_score": comp["score"],
            "competition_analysis_evidence": comp["evidence"],
            "risk_score": risk["score"],
            "risk_analysis_evidence": risk["evidence"]
        }
    except Exception as e:
        print(f"❌ CompAnalysis 중 오류: {e}")
        return {
            "competition_score": 0, "competition_analysis_evidence": f"분석 오류: {e}",
            "risk_score": 0, "risk_analysis_evidence": f"분석 오류: {e}"
        }


def node_invest_decision(state: AgentState) -> Dict: 
    w = {
        "technology_score": 0.20,
        "learning_effectiveness_score": 0.15,
        "market_score": 0.20,
        "competition_score": 0.15,
        "risk_score": 0.10,
        "growth_potential_score": 0.20
    }
    total = sum(state.get(k, 0)*w[k] for k in w.keys())
    judge = "투자" if total >= INVESTMENT_THRESHOLDS["투자"] else "보류"
    
    print(f"[bold yellow]💡 InvestDecision:[/] 총점 {total:.1f} → {judge}")
    
    return {"final_judge": judge}


def node_hold(state: AgentState) -> Dict: 
    """보류 시 다음 후보 반복 (단순히 SelectOne으로 루프)"""
    print("[bold magenta]⏸️ Hold: 다음 후보로 이동[/]")
    return {} # 빈 딕셔너리(유효한 업데이트)를 반환


def node_report(state: AgentState) -> Dict: 
    c = state.get("current_candidate") or {}
    name = c.get("name", "Unknown")
    md = f"""
[스타트업] {name}

기술력: {state.get('technology_score',0)}점 — {state.get('technology_analysis_evidence','')}
학습 성과: {state.get('learning_effectiveness_score',0)}점 — {state.get('learning_effectiveness_analysis_evidence','')}
시장성: {state.get('market_score',0)}점 — {state.get('market_analysis_evidence','')}
경쟁력: {state.get('competition_score',0)}점 — {state.get('competition_analysis_evidence','')}
리스크(안전도): {state.get('risk_score',0)}점 — {state.get('risk_analysis_evidence','')}
성장 가능성: {state.get('growth_potential_score',0)}점 — {state.get('growth_potential_analysis_evidence','')}

▶ 최종 판단: {state.get('final_judge','보류')}
""".strip()

    safe_name = re.sub(r'[^0-9A-Za-z가-힣_\-]+','_',name) or "Unknown"
    pdf_path = f"{safe_name}_invest_report.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = [Paragraph(md.replace("\n","<br/>"), styles["Normal"]), Spacer(1,12)]
    doc.build(story)
    
    print(f"[bold green]📝 Report:[/] PDF 생성 완료 → {pdf_path}")
    
    return {"report": md, "pdf_path": pdf_path}


# =========================
# 그래프 구성 (Flowchart 그대로)
# =========================
def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("WebCrawling", node_web_crawling)
    graph.add_node("Filtering", node_filtering)
    graph.add_node("SelectOne", node_select_one)
    graph.add_node("TechSummary", node_tech_summary)
    graph.add_node("MarketEval", node_market_eval)
    graph.add_node("CompAnalysis", node_comp_analysis)
    graph.add_node("InvestDecision", node_invest_decision)
    graph.add_node("Hold", node_hold)
    graph.add_node("Report", node_report)

    graph.set_entry_point("WebCrawling")
    graph.add_edge("WebCrawling", "Filtering")

    def guard_filter(state: AgentState):
        return "SelectOne" if state.get("filtered") else "End"
    graph.add_conditional_edges("Filtering", guard_filter, {"SelectOne":"SelectOne", "End": END})
    
    def guard_select(state: AgentState):
        return "Analyze" if state.get("current_candidate") else "End"
    graph.add_conditional_edges("SelectOne", guard_select, {"Analyze": "TechSummary", "End": END})

    graph.add_edge("TechSummary", "MarketEval")
    graph.add_edge("MarketEval", "CompAnalysis")
    graph.add_edge("CompAnalysis", "InvestDecision")

    def guard_decision(state: AgentState):
        return "Report" if state.get("final_judge") == "투자" else "Hold"
    graph.add_conditional_edges("InvestDecision", guard_decision, {"Report":"Report", "Hold":"Hold"})

    graph.add_edge("Hold", "SelectOne") 
    graph.add_edge("Report", END)

    # [💡💡💡] SqliteSaver 관련 코드를 모두 삭제합니다.
    # memory = SqliteSaver.in_memory() 
    
    return graph.compile() # <--- checkpointer가 제거되었습니다.


# =========================
# 실행
# =========================
if __name__ == "__main__":
    
    app = build_app()
    init: Dict = {} 

    # [💡💡💡] checkpointer가 없으므로 thread_id가 필요 없습니다.
    # thread_id = f"edtech_run_{int(time.time())}"
    print(f"🚀 실행 시작")

    try:
        # [💡💡💡] config=... 부분이 제거되었습니다.
        final_state = app.invoke(init)
        
        print("\n=== 최종 결과 ===")
        if final_state.get("report"):
            print(final_state["report"])
            print("PDF:", final_state["pdf_path"])
        else:
            print("보고서 없이 종료됨(후보 없음 또는 모두 보류)")
            
    except Exception as e:
        print(f"\n❌ 그래프 실행 중 치명적 오류 발생: {e}")
        traceback.print_exc()