"""
agents/risk_agent.py
리스크 분석 Agent
"""
from langchain_core.prompts import ChatPromptTemplate
from .base import AgentState, EVALUATION_CRITERIA, llm, extract_score, get_web_context


def risk_agent(state: AgentState) -> AgentState:
    """Agent 6: 리스크 분석"""
    print("\n⚠️ [Agent 6] 리스크 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["risk"]
    context = get_web_context(startup_name, "리스크 이슈 문제")
    
    prompt = ChatPromptTemplate.from_template("""
교육 스타트업 '{startup_name}'의 리스크를 평가하세요.

**평가 기준 (각 항목 0-10점, 점수가 높을수록 리스크가 낮음):**
{checklist}

**참고 자료:**
{context}

**출력 형식:**
각 항목별로:
- 점수 (0-10점, 높을수록 리스크 낮음)
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
    
    state["risk_score"] = score
    
    print(f"✅ [Agent 6] 완료 - 리스크 점수: {score} (높을수록 안전)")
    return state