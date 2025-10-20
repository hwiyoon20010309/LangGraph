"""
agents/risk_agent.py
리스크 분석 Agent
"""
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from .base import AgentState, EVALUATION_CRITERIA, extract_score, get_web_context
from agents.llm_factory import get_llm  # ✅ llm은 공장 함수로 생성(ORG/PROJECT 대응)

MAX_CONTEXT_CHARS = 9000  # ✅ 과도한 프롬프트 길이 방지

def _safe_extract_total_score(text: str, default: int = 60) -> int:
    """총점(0-100)을 안전하게 추출하고 범위 클램프."""
    try:
        score = extract_score(text)  # 사용자가 만든 파서
        if score is None:
            return default
        # 0~100 범위 보정
        score = int(score)
        if score < 0: score = 0
        if score > 100: score = 100
        return score
    except Exception:
        return default

def risk_agent(state: AgentState) -> Dict:
    """Agent 6: 리스크 분석"""
    print("\n⚠️ [Agent 6] 리스크 분석 시작...")

    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["risk"]

    # ✅ 웹 컨텍스트 수집 + 길이 제한
    raw_context = get_web_context(startup_name, "리스크 이슈 문제") or ""
    context = raw_context[:MAX_CONTEXT_CHARS] if raw_context else "관련 공개 자료가 충분치 않습니다."

    # ✅ 시스템 메시지로 역할 고정 + 출력 규격 강조
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 VC의 리스크 분석가입니다. 근거 기반으로 간결하게 작성하고, "
         "요청된 출력 형식을 반드시 지키세요. 하이루머/추측은 금지."),
        ("human", """
교육 스타트업 '{startup_name}'의 리스크를 평가하세요.

**평가 기준 (각 항목 0-10점, 점수가 높을수록 리스크가 낮음):**
{checklist}

**참고 자료:**
{context}

**출력 형식(반드시 준수):**
각 항목별로:
- 점수 (0-10점, 높을수록 리스크 낮음)
- 근거 (가능하면 URL 포함)

마지막 줄에 **총점: [0-100 숫자]** 만 한 줄로 표기하세요.
""")
    ])

    # ✅ ORG/PROJECT 헤더 포함된 LLM 생성 (401 방지)
    llm = get_llm(model="gpt-4o-mini", temperature=0.1)

    try:
        response = (prompt | llm).invoke({
            "startup_name": startup_name,
            "checklist": "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist)),
            "context": context
        })
        analysis = getattr(response, "content", str(response))
        score100 = _safe_extract_total_score(analysis)  # 0~100
        # 👉 메인에서 0~5/0~10 스케일이면 여기서 변환해도 됨. 예: 100점→5점 환산
        #    risk는 "높을수록 안전"이므로 100점을 5점으로 스케일 다운:
        risk_score_10 = round(score100 / 10)  # 0~10
        print(f"✅ [Agent 6] 완료 - 리스크 총점(100기준): {score100} → 10점 스케일: {risk_score_10}")
        return {
            "risk_score": risk_score_10,
            # 필요시 증거 저장:
            # "risk_analysis_evidence": analysis[:2000],
        }

    except Exception as e:
        # ✅ 인증/네트워크/타임아웃 등 예외 폴백
        print(f"❌ [Agent 6] 호출 오류: {e}")
        # 보수적 기본값(중간치) 반환
        return {
            "risk_score": 3
        }
