"""
main.py
투자 심사 시스템 메인 실행 파일
"""
from langgraph.graph import StateGraph, END
from agents.base import AgentState
from agents.technology_agent import technology_agent
from agents.learning_effectiveness_agent import learning_effectiveness_agent
from agents.market_agent import market_agent
from agents.competition_agent import competition_agent
from agents.growth_potential_agent import growth_potential_agent
from agents.risk_agent import risk_agent
from agents.judge_agent import comprehensive_judge_agent
from agents.report_agent import report_generation_agent


def build_agent_workflow():
    """독립적인 Agent 기반 워크플로우 구축"""
    
    workflow = StateGraph(AgentState)
    
    # Agent 노드 추가
    workflow.add_node("technology", technology_agent)
    workflow.add_node("learning", learning_effectiveness_agent)
    workflow.add_node("market", market_agent)
    workflow.add_node("competition", competition_agent)
    workflow.add_node("growth", growth_potential_agent)
    workflow.add_node("risk", risk_agent)
    workflow.add_node("judge", comprehensive_judge_agent)
    workflow.add_node("report", report_generation_agent)
    
    # 순차 실행 플로우
    workflow.set_entry_point("technology")
    workflow.add_edge("technology", "learning")
    workflow.add_edge("learning", "market")
    workflow.add_edge("market", "competition")
    workflow.add_edge("competition", "growth")
    workflow.add_edge("growth", "risk")
    workflow.add_edge("risk", "judge")
    workflow.add_edge("judge", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def run_investment_analysis(startup_name: str):
    """투자 분석 실행"""
    
    print("=" * 70)
    print(f"🎯 투자 심사 시작: {startup_name}")
    print("=" * 70)
    
    # 초기 State
    initial_state: AgentState = {
        "startup_name": startup_name,
        "technology_score": 0,
        "technology_analysis_evidence": "",
        "learning_effectiveness_score": 0,
        "learning_effectiveness_analysis_evidence": "",
        "market_score": 0,
        "market_analysis_evidence": "",
        "competition_score": 0,
        "competition_analysis_evidence": "",
        "growth_potential_score": 0,
        "growth_potential_analysis_evidence": "",
        "risk_score": 0,
        "final_judge": "보류",
        "report": "",
        "pdf_path": ""
    }
    
    # Agent 워크플로우 실행
    agent = build_agent_workflow()
    final_state = agent.invoke(initial_state)
    
    # 최종 결과 출력
    print("\n" + "=" * 70)
    print("📊 최종 결과")
    print("=" * 70)
    print(f"스타트업: {final_state['startup_name']}")
    print(f"투자 결정: {final_state['final_judge']}")
    print(f"\n항목별 점수:")
    print(f"  🔧 기술력: {final_state['technology_score']}")
    print(f"  📚 학습효과: {final_state['learning_effectiveness_score']}")
    print(f"  💰 시장성: {final_state['market_score']}")
    print(f"  ⚔️ 경쟁력: {final_state['competition_score']}")
    print(f"  🚀 성장가능성: {final_state['growth_potential_score']}")
    print(f"  ⚠️ 리스크: {final_state['risk_score']}")
    print(f"\n📄 보고서: {final_state['pdf_path']}")
    print("=" * 70)
    
    return final_state


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
        preview = result["report"][:1000]
        print(preview + "..." if len(result["report"]) > 1000 else preview)
        print("=" * 70)