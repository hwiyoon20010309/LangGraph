"""
agents/judge_agent.py
종합 판단 Agent
"""
from langchain_core.prompts import ChatPromptTemplate
from .base import AgentState, llm


def comprehensive_judge_agent(state: AgentState) -> AgentState:
    """Agent 7: 종합 판단 - State의 모든 점수를 기반으로 투자 결정"""
    print("\n⚖️ [Agent 7] 종합 판단 시작...")
    
    # State에서 점수 수집
    tech = state["technology_score"]
    learning = state["learning_effectiveness_score"]
    market = state["market_score"]
    competition = state["competition_score"]
    growth = state["growth_potential_score"]
    risk = state["risk_score"]
    
    # 가중 평균 계산
    weights = {
        "tech": 0.20,
        "learning": 0.20,
        "market": 0.25,
        "competition": 0.15,
        "growth": 0.10,
        "risk": 0.10
    }
    
    total_score = int(
        tech * weights["tech"] +
        learning * weights["learning"] +
        market * weights["market"] +
        competition * weights["competition"] +
        growth * weights["growth"] +
        risk * weights["risk"]
    )
    
    prompt = ChatPromptTemplate.from_template("""
다음 점수를 바탕으로 투자 결정을 내리세요:

**점수 현황:**
- 기술력: {tech}/100 (가중치 20%)
- 학습효과: {learning}/100 (가중치 20%)
- 시장성: {market}/100 (가중치 25%)
- 경쟁력: {competition}/100 (가중치 15%)
- 성장가능성: {growth}/100 (가중치 10%)
- 리스크: {risk}/100 (가중치 10%, 높을수록 안전)

**가중 평균 총점: {total}/100**

**판단 기준:**
- 총점 70 이상 AND 모든 항목 50 이상 → "투자"
- 총점 50-69 OR 일부 항목 50 미만 → "보류"
- 총점 50 미만 → "보류"

**출력 형식:**
결정만 출력: 투자 또는 보류
""")
    
    response = (prompt | llm).invoke({
        "tech": tech,
        "learning": learning,
        "market": market,
        "competition": competition,
        "growth": growth,
        "risk": risk,
        "total": total_score
    })
    
    # 투자 결정 추출
    decision_text = response.content.strip()
    decision = "투자" if "투자" in decision_text and "보류" not in decision_text else "보류"
    
    print(f"✅ [Agent 7] 완료 - 최종 결정: {decision} (총점: {total_score})")
    
    # 자신의 필드만 반환
    return {
        "final_judge": decision
    }