"""
agents/growth_potential_agent.py
성장 가능성 분석 Agent
"""
from langchain_core.prompts import ChatPromptTemplate
from .base import AgentState, EVALUATION_CRITERIA, llm, extract_score, get_web_context


def growth_potential_agent(state: AgentState) -> AgentState:
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
    
    print(f"✅ [Agent 5] 완료 - 성장가능성 점수: {score}")
    
    # 자신의 필드만 반환
    return {
        "growth_potential_score": score,
        "growth_potential_analysis_evidence": analysis
    }