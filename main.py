"""
main.py
투자 심사 시스템 메인 실행 파일 (병렬 처리 최적화)
"""
from typing import Sequence
from langgraph.graph import StateGraph, END, START
from agents.base import AgentState
from agents.technology_agent import technology_agent
from agents.learning_effectiveness_agent import learning_effectiveness_agent
from agents.market_agent import market_agent
from agents.competition_agent import competition_agent
from agents.growth_potential_agent import growth_potential_agent
from agents.risk_agent import risk_agent
from agents.judge_agent import comprehensive_judge_agent
from agents.report_agent import report_generation_agent
import time


def start_node(state: AgentState) -> dict:
    """시작 노드 - 병렬 실행을 위한 진입점"""
    print("\n🚀 분석 시작 - 6개 Agent 병렬 실행 준비")
    print("-" * 70)
    # 아무것도 업데이트하지 않음 (빈 딕셔너리 반환)
    return {}


def route_to_parallel_agents(state: AgentState) -> Sequence[str]:
    """
    병렬로 실행할 Agent 노드 목록 반환
    이 함수가 반환하는 노드들이 모두 병렬로 실행됨
    """
    return ["technology", "learning", "market", "competition", "growth", "risk"]


def build_agent_workflow():
    """
    병렬 Agent 기반 워크플로우 구축 (Fan-out & Fan-in)
    
    구조:
    START → start → [6개 분석 Agent 병렬 실행] → Judge → Report → END
    """
    
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("start", start_node)
    workflow.add_node("technology", technology_agent)
    workflow.add_node("learning", learning_effectiveness_agent)
    workflow.add_node("market", market_agent)
    workflow.add_node("competition", competition_agent)
    workflow.add_node("growth", growth_potential_agent)
    workflow.add_node("risk", risk_agent)
    workflow.add_node("judge", comprehensive_judge_agent)
    workflow.add_node("report", report_generation_agent)
    
    # ========================================
    # Fan-out: start → 6개 Agent 병렬 분기
    # ========================================
    workflow.add_edge(START, "start")
    
    # 병렬로 실행할 agent 목록
    parallel_agents = ["technology", "learning", "market", "competition", "growth", "risk"]
    
    # start 노드에서 6개 agent로 조건부 분기 (모두 병렬 실행)
    workflow.add_conditional_edges(
        "start",
        route_to_parallel_agents,
        parallel_agents,  # 가능한 경로 목록
    )
    
    # ========================================
    # Fan-in: 모든 병렬 Agent → Judge로 수렴
    # ========================================
    for agent in parallel_agents:
        workflow.add_edge(agent, "judge")
    
    # ========================================
    # 순차 실행: Judge → Report → END
    # ========================================
    workflow.add_edge("judge", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def run_investment_analysis(startup_name: str):
    """투자 분석 실행 (병렬 처리)"""
    
    print("=" * 70)
    print(f"🎯 투자 심사 시작: {startup_name}")
    print("🚀 Fan-out & Fan-in 병렬 처리 모드")
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
    
    # 시작 시간 측정
    start_time = time.time()
    
    # Agent 워크플로우 실행
    agent = build_agent_workflow()
    final_state = agent.invoke(initial_state)
    
    # 종료 시간 측정
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 최종 결과 출력
    print("\n" + "=" * 70)
    print("📊 최종 결과")
    print("=" * 70)
    print(f"스타트업: {final_state['startup_name']}")
    print(f"투자 결정: {final_state['final_judge']}")
    print(f"⏱️  총 실행 시간: {execution_time:.2f}초")
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


def compare_performance(startup_name: str):
    """순차 vs 병렬 성능 비교 (테스트용)"""
    print("\n" + "🔬 " * 20)
    print("성능 비교 테스트: 순차 실행 vs 병렬 실행")
    print("🔬 " * 20 + "\n")
    
    # 순차 실행 워크플로우
    def build_sequential_workflow():
        workflow = StateGraph(AgentState)
        workflow.add_node("technology", technology_agent)
        workflow.add_node("learning", learning_effectiveness_agent)
        workflow.add_node("market", market_agent)
        workflow.add_node("competition", competition_agent)
        workflow.add_node("growth", growth_potential_agent)
        workflow.add_node("risk", risk_agent)
        workflow.add_node("judge", comprehensive_judge_agent)
        workflow.add_node("report", report_generation_agent)
        
        workflow.add_edge(START, "technology")
        workflow.add_edge("technology", "learning")
        workflow.add_edge("learning", "market")
        workflow.add_edge("market", "competition")
        workflow.add_edge("competition", "growth")
        workflow.add_edge("growth", "risk")
        workflow.add_edge("risk", "judge")
        workflow.add_edge("judge", "report")
        workflow.add_edge("report", END)
        return workflow.compile()
    
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
    
    # 순차 실행
    print("🐌 순차 실행 모드 테스트 중...")
    seq_agent = build_sequential_workflow()
    seq_start = time.time()
    seq_agent.invoke(initial_state)
    seq_time = time.time() - seq_start
    print(f"   완료: {seq_time:.2f}초")
    
    # 병렬 실행
    print("\n🚀 병렬 실행 모드 테스트 중...")
    par_agent = build_agent_workflow()
    par_start = time.time()
    par_agent.invoke(initial_state)
    par_time = time.time() - par_start
    print(f"   완료: {par_time:.2f}초")
    
    # 결과 비교
    speedup = seq_time / par_time if par_time > 0 else 0
    improvement = ((seq_time - par_time) / seq_time) * 100 if seq_time > 0 else 0
    
    print("\n" + "=" * 70)
    print("📈 성능 비교 결과")
    print("=" * 70)
    print(f"🐌 순차 실행: {seq_time:.2f}초")
    print(f"🚀 병렬 실행: {par_time:.2f}초")
    print(f"⚡ 속도 향상: {speedup:.2f}배")
    print(f"📊 성능 개선: {improvement:.1f}%")
    print(f"⏰ 절약 시간: {seq_time - par_time:.2f}초")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # 성능 비교 모드
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        startup = input("성능 비교할 교육 스타트업 이름: ").strip()
        if startup:
            compare_performance(startup)
        sys.exit(0)
    
    # 일반 실행 모드
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