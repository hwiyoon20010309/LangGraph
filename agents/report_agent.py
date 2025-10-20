"""
agents/report_agent.py
보고서 생성 Agent
"""
import os
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from .base import AgentState, llm


def report_generation_agent(state: AgentState) -> AgentState:
    """Agent 8: 최종 보고서 생성 - State 기반"""
    print("\n📝 [Agent 8] 보고서 생성 시작...")
    
    prompt = ChatPromptTemplate.from_template("""
# 투자 심사 보고서

## 기본 정보
- 스타트업: {startup_name}
- 작성일: {date}
- 최종 결정: **{decision}**

## 항목별 점수
| 항목 | 점수 | 가중치 |
|------|------|--------|
| 기술력 | {tech}/100 | 20% |
| 학습효과 | {learning}/100 | 20% |
| 시장성 | {market}/100 | 25% |
| 경쟁력 | {competition}/100 | 15% |
| 성장가능성 | {growth}/100 | 10% |
| 리스크 | {risk}/100 | 10% |

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

## 최종 종합 결론
위 분석 결과를 바탕으로 SWOT 분석과 투자 권고사항을 작성하세요.
""")
    
    response = (prompt | llm).invoke({
        "startup_name": state["startup_name"],
        "date": datetime.now().strftime("%Y년 %m월 %d일"),
        "decision": state["final_judge"],
        "tech": state["technology_score"],
        "learning": state["learning_effectiveness_score"],
        "market": state["market_score"],
        "competition": state["competition_score"],
        "growth": state["growth_potential_score"],
        "risk": state["risk_score"],
        "tech_evidence": state["technology_analysis_evidence"],
        "learning_evidence": state["learning_effectiveness_analysis_evidence"],
        "market_evidence": state["market_analysis_evidence"],
        "competition_evidence": state["competition_analysis_evidence"],
        "growth_evidence": state["growth_potential_analysis_evidence"]
    })
    
    state["report"] = response.content
    
    # 파일 저장
    output_dir = "investment_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{state['startup_name']}_투자분석_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(state["report"])
    
    print(f"✅ [Agent 8] 완료 - 보고서 저장: {filepath}")
    
    # 자신의 필드만 반환
    return {
        "report": response.content,
        "pdf_path": filepath
    }