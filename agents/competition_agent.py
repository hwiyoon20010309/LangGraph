"""
agents/competition_agent.py
경쟁력 분석 Agent
"""
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain_tavily import TavilySearch
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults as TavilySearch
from .base import AgentState, EVALUATION_CRITERIA, llm, extract_score


def competition_agent(state: AgentState) -> AgentState:
    """Agent 4: 경쟁력 분석"""
    print("\n⚔️ [Agent 4] 경쟁력 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["competition"]
    
    try:
        search = TavilySearch(max_results=15)
        results = search.invoke(f"{startup_name} 경쟁사 비교")
        # TavilySearch는 문자열을 직접 반환하거나 리스트를 반환할 수 있음
        if isinstance(results, str):
            context = results
        elif isinstance(results, list):
            context = "\n".join([f"- {r.get('title', r.get('content', str(r)))}" for r in results])
        else:
            context = str(results)
    except Exception as e:
        context = f"검색 실패: {str(e)}"
    
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
    state["competition_analysis_evidence"] = analysis
    
    print(f"✅ [Agent 4] 완료 - 경쟁력 점수: {score}")
    return state