"""
agents/technology_agent.py
기술력 분석 Agent
"""
from langchain_core.prompts import ChatPromptTemplate
from .base import AgentState, EVALUATION_CRITERIA, llm, extract_score, get_web_context


def technology_agent(state: AgentState) -> AgentState:
    """Agent 1: 기술력 분석"""
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
    state["technology_analysis_evidence"] = analysis
    
    print(f"✅ [Agent 1] 완료 - 기술력 점수: {score}")
    return state