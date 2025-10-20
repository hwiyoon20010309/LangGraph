# -*- coding: utf-8 -*-
"""
EdTech 투자 파이프라인 (A: 목록 생성 -> 전체 평가/랭킹 -> 순위 기반 B-H 검증)

흐름:
1. Agent A: 여러 쿼리로 Tavily 검색 -> 이름 추출 -> AI 필터링 -> 초기 목록 생성 (CSV 저장)
2. EvaluateAll & Rank: 초기 목록의 모든 스타트업 상세 정보 검색 -> 6기준 AI 평가 -> 총점 계산 -> 순위 매기기 (CSV 저장 & State 저장)
3. Select Ranked Startup: 순위 목록에서 다음 순서의 스타트업 선택
4. Agents B-H: 선택된 스타트업 순차 검증 (pass/fail)
5. 라우팅: fail 시 다음 순위 선택, 모두 pass 시 성공, 목록 소진 시 종료

필요:
- pip install langgraph langchain langchain-openai langchain-community tiktoken requests beautifulsoup4 python-dotenv rich langchain-opentutorial pandas
- .env:
    OPENAI_API_KEY=sk-...
    TAVILY_API_KEY=tvly-...
"""

import os
import re
import requests
import pandas as pd
import random
import json
import time
from typing import List, Set, Dict, Optional, TypedDict, Literal
from dotenv import load_dotenv
from openai import OpenAI
from rich import print

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.tools.tavily import TavilySearch

# === ① 환경 변수 로드 ===
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("🚨 Tavily API key missing (.env).")
if not OPENAI_API_KEY:
    raise ValueError("🚨 OpenAI API key missing (.env).")

client = OpenAI(api_key=OPENAI_API_KEY)
TAVILY_URL = "https://api.tavily.com/search"
# CSV 파일명 정의
INITIAL_STARTUP_CSV = "ai_filtered_startups.csv"
RANKED_CSV_FILE = "ranked_startup_evaluations.csv"

# === ②-1. 스타트업 목록 생성을 위한 설정 ===
AGGREGATION_QUERIES = [
    "AI 교육 스타트업 목록 (예: Riiid, Mathpresso, Sana Labs, Squirrel AI 등)",
    "EdTech 분야에서 인공지능(AI)을 활용하는 주요 스타트업",
    "Top AI education or EdTech startups 2025 funding news",
    "AI tutoring platform startups using adaptive learning investment",
]
NAME_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]*){0,2})\b"
)
STOPWORDS = {
    "AI", "Labs", "Learning", "Education", "EdTech", "Systems", "Company", "Group",
    "Technology", "Technologies", "Platform", "Startup", "News", "Report", "Top",
    "Software", "Tools", "Adaptive", "Artificial", "Intelligence", "Market",
    "Overview", "Trend", "Global", "Data", "Model", "Classroom", "Program", "School",
    "South", "Korea", "Best", "List", "World",
}

# === ②-2. 상세 평가를 위한 기준 ===
EVALUATION_CRITERIA = {
    "technology": [
        "제품이 교육 문제를 명확하게 해결하는가?", "AI/ML 기술 활용도가 높은가?",
        "기술의 혁신성과 차별화가 있는가?", "기술적 구현 가능성이 높은가?",
        "시스템의 확장 가능성이 있는가?", "기술 안정성과 보안이 확보되어 있는가?",
        "데이터 기반 학습 최적화가 가능한가?", "API 연동 및 확장성이 뛰어난가?",
        "기술 문서화가 잘 되어 있는가?", "오픈소스 활용 및 커뮤니티 기여도가 있는가?"
    ],
    "learning_effectiveness": [
        "학습 성과 측정 지표가 명확한가?", "학습자 만족도가 높은가?",
        "학습 완료율이 우수한가?", "학습 효과 검증 사례가 있는가?",
        "개인화 학습 지원이 가능한가?", "학습 데이터 분석 및 피드백 제공이 되는가?",
        "교사/강사 지원 도구가 있는가?", "학습자 참여도 향상 방안이 있는가?",
        "콘텐츠 품질이 우수한가?", "학습 경로 추천이 효과적인가?"
    ],
    "market": [
        "타겟 교육 시장 규모가 큰가?", "시장 성장률이 높은가?",
        "수익 모델이 명확하고 실현 가능한가?", "고객 기반(B2B/B2C)이 확보되어 있는가?",
        "가격 전략이 합리적인가?", "시장 진입 전략이 구체적인가?",
        "고객 획득 비용(CAC)이 적절한가?", "생애 가치(LTV)가 높은가?",
        "파트너십 확보 가능성이 있는가?", "글로벌 시장 진출 가능성이 있는가?"
    ],
    "competition": [
        "경쟁사 대비 명확한 차별화 요소가 있는가?", "시장 진입 장벽이 존재하는가?",
        "경쟁 우위(특허, 기술, 네트워크)가 있는가?", "브랜드 인지도가 형성되어 있는가?",
        "고객 충성도가 높은가?", "선점 효과(First Mover)가 있는가?",
        "네트워크 효과가 작동하는가?", "전환 비용(Switching Cost)이 높은가?",
        "경쟁사 대비 가성비가 우수한가?", "지속 가능한 경쟁력이 있는가?"
    ],
    "growth_potential": [
        "시장 확장 가능성이 큰가?", "제품 다각화 계획이 있는가?",
        "글로벌 진출 전략이 구체적인가?", "파트너십 확대 기회가 있는가?",
        "인수합병(M&A) 가능성이 있는가?", "IPO 가능성이 있는가?",
        "스케일업을 위한 인프라가 준비되어 있는가?", "투자 유치 이력이 있는가?",
        "성장 로드맵이 명확한가?", "10배 성장(10x Growth) 가능성이 있는가?"
    ],
    "risk": [
        "재무 리스크가 낮은가?", "법적/규제 리스크가 낮은가?", "기술 리스크가 낮은가?",
        "시장 리스크가 낮은가?", "경영진 리스크가 낮은가?", "운영 리스크가 낮은가?",
        "평판 리스크가 낮은가?", "경쟁 리스크가 낮은가?",
        "파트너십 의존도 리스크가 낮은가?", "확장성 리스크가 낮은가?"
    ]
}

# === ③ LangGraph 상태 스키마 ===
class AgentState(TypedDict, total=False):
    initial_candidates: List[str]
    ranked_evaluations: List[Dict]
    current_rank_index: int
    current_startup_details: Dict
    analysis_b: str
    analysis_c: str
    analysis_d: str
    analysis_e: str
    analysis_f: str
    analysis_g: str
    analysis_h: str
    last_decision: Literal["continue", "reject"]

# === ④ Helper 함수들 ===

def tavily_search_for_aggregation(query: str, max_results: int = 40) -> dict:
    """스타트업 목록 생성을 위한 Tavily 검색 함수"""
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "query": query, "max_results": min(max_results, 50),
        "include_answer": True, "search_depth": "advanced",
    }
    try:
        res = requests.post(TAVILY_URL, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ (목록 생성) Tavily 검색 실패({query}): {e}")
        return {}
    except Exception as e:
         print(f"❌ (목록 생성) Tavily 처리 중 오류({query}): {e}")
         return {}

def extract_candidate_names(*texts: str) -> Set[str]:
    """텍스트에서 스타트업 이름 후보를 추출하는 함수"""
    candidates: Set[str] = set()
    for text in texts:
        if not text: continue
        for match in NAME_PATTERN.findall(text):
            name = match.strip()
            # 필터링 조건
            if (len(name) < 3 or name.upper() in STOPWORDS or
                not any(c.islower() for c in name if c.isalpha()) or
                len(name.split()) > 3 or
                not re.match(r"^[A-Z]", name) or
                re.search(r'\d{4}', name)):
                continue
            candidates.add(name)
    return candidates

def ai_filter_startups(candidates: List[str]) -> List[str]:
    """GPT를 사용하여 후보 목록에서 실제 기업명만 필터링하는 함수"""
    if not candidates: return []
    prompt = f"""다음 리스트에서 실제 'AI 기반 교육(EdTech) 스타트업' 또는 관련 기업의 이름만 정확히 추출해주세요. 뉴스 제목의 일부, 일반 명사, 기술 용어, 인물/도시 이름, 보고서 제목 등은 모두 제외하고 오직 회사 이름만 남겨야 합니다. 결과는 회사 이름만 한 줄에 하나씩 나열해주세요. 중복은 제거해주세요.
리스트: {', '.join(candidates)}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        text = response.choices[0].message.content.strip()
        return sorted(list(set(line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 1)))
    except Exception as e:
        print(f"⚠️ AI 필터 실패: {e}. 원본 후보 반환.")
        return sorted(list(set(candidates)))

def get_startup_context_for_eval(startup_name: str, max_results: int = 7) -> str:
    """상세 평가를 위해 특정 스타트업의 컨텍스트를 Tavily에서 검색하는 함수"""
    print(f"  🔍 '{startup_name}' 상세 정보 검색 중 (Tavily)...")
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    query = f"{startup_name} EdTech company overview funding technology business model market size competition recent news"
    payload = { "query": query, "max_results": max_results, "include_answer": True, "search_depth": "advanced" }
    try:
        res = requests.post(TAVILY_URL, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        data = res.json()
        context_parts = []
        if data.get("answer"): context_parts.append(f"Tavily 요약:\n{data['answer']}")
        if data.get("results"):
            for i, result in enumerate(data["results"]):
                title = result.get("title", "N/A")
                content = result.get("content", "N/A")
                if content and len(content) > 50:
                    context_parts.append(f"\n출처 {i+1} ({title}):\n{content}")
        if not context_parts:
             print(f"  ⚠️ '{startup_name}'에 대한 상세 검색 결과가 없습니다.")
             return "검색된 상세 정보 없음."
        print(f"  ✅ '{startup_name}' 상세 정보 검색 완료 ({len(context_parts)}개 출처).")
        full_context = "\n".join(context_parts)
        return full_context[:15000] # 토큰 제한 고려
    except requests.exceptions.RequestException as e:
        print(f"  ❌ (상세 검색) Tavily 네트워크 오류: {e}")
        return f"Tavily 상세 검색 실패: {e}"
    except Exception as e:
        print(f"  ❌ (상세 검색) Tavily 처리 중 오류: {e}")
        return f"Tavily 상세 검색 실패: {e}"

def evaluate_startup_with_ai(startup_name: str, context: str, criteria: Dict[str, List[str]]) -> Optional[Dict]:
    """GPT를 사용하여 스타트업 상세 평가를 수행하는 함수"""
    print(f"  🤖 '{startup_name}' 상세 평가 시작 (GPT-4o-mini)...")
    criteria_prompt_text = ""
    for category, questions in criteria.items():
        criteria_prompt_text += f"\n### {category.upper()} 평가 기준:\n" + "\n".join(f"- {q}" for q in questions)
    prompt = f"""
당신은 매우 꼼꼼한 EdTech VC 투자 심사역입니다. 주어진 스타트업 '{startup_name}'에 대한 정보(Context)를 바탕으로 다음 평가 기준들을 **종합적으로 고려**하여 분석해주세요.

**Context:**
{context}
---
**평가 기준:**
{criteria_prompt_text}
---
**분석 요청:**
위 Context와 평가 기준들을 바탕으로, **각 6가지 카테고리(technology, learning_effectiveness, market, competition, growth_potential, risk)별**로 스타트업이 얼마나 우수한지 **종합적인 분석**을 1-2 문장으로 작성하고, **1점에서 5점 사이의 점수**를 매겨주세요. (5점이 가장 우수함. 단, risk는 점수가 높을수록 리스크가 낮음을 의미)

**출력 형식 (오직 JSON 객체만 출력, 다른 설명 절대 금지):**
{{
  "startup_name": "{startup_name}",
  "evaluation_summary": {{
    "technology": {{ "analysis": "...", "score": 점수(1-5) }},
    "learning_effectiveness": {{ "analysis": "...", "score": 점수(1-5) }},
    "market": {{ "analysis": "...", "score": 점수(1-5) }},
    "competition": {{ "analysis": "...", "score": 점수(1-5) }},
    "growth_potential": {{ "analysis": "...", "score": 점수(1-5) }},
    "risk": {{ "analysis": "...", "score": 점수(1-5) }}
  }},
  "overall_assessment": "종합 투자 의견..."
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        result_json = response.choices[0].message.content
        print(f"  ✅ '{startup_name}' 상세 평가 완료.")
        try:
            parsed_result = json.loads(result_json)
            if "evaluation_summary" in parsed_result and "overall_assessment" in parsed_result:
                 # 점수가 숫자인지 확인 및 변환 (LLM이 가끔 문자열로 줄 수 있음)
                 for category_data in parsed_result.get("evaluation_summary", {}).values():
                     if isinstance(category_data.get("score"), str):
                         try:
                             category_data["score"] = int(category_data["score"])
                         except ValueError:
                              print(f"⚠️ '{startup_name}' {category_data} 점수 변환 오류 -> 0점 처리")
                              category_data["score"] = 0
                     elif not isinstance(category_data.get("score"), (int, float)):
                         category_data["score"] = 0 # 숫자가 아니면 0점 처리

                 return parsed_result
            else:
                 print(f"  ❌ AI 평가 JSON 구조 오류. 원본 응답:\n{result_json}")
                 return None
        except json.JSONDecodeError as e:
            print(f"  ❌ AI 평가 JSON 파싱 실패: {e}. 원본 응답:\n{result_json}")
            return None
    except Exception as e:
        print(f"  ❌ AI 평가 중 오류 발생: {e}")
        return None

# === ⑤ LangGraph 노드 함수 정의 ===

def node_agent_a_generate_list(state: AgentState) -> Dict:
    """Agent A: 초기 스타트업 목록 생성 및 CSV 저장"""
    print("\n[bold blue]=== 🚀 1단계: AI EdTech 스타트업 목록 생성 시작 ===[/]")
    all_candidates: Set[str] = set()
    for q in AGGREGATION_QUERIES:
        print(f"  🔍 Searching: {q}")
        data = tavily_search_for_aggregation(q)
        if not data: continue
        all_candidates.update(extract_candidate_names(data.get("answer", "")))
        for r in data.get("results", []):
            all_candidates.update(extract_candidate_names(r.get("title", ""), r.get("content", "")))

    print(f"  🧩 1차 추출된 후보 수: {len(all_candidates)}")
    if not all_candidates:
         print("  ❌ 1차 추출된 후보가 없습니다.")
         return {"initial_candidates": []}

    print("  🤖 AI 필터링 진행 중...")
    filtered_startups = ai_filter_startups(sorted(list(all_candidates)))

    if filtered_startups:
        df = pd.DataFrame(filtered_startups, columns=["startup_name"])
        df.to_csv(INITIAL_STARTUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  ✅ 최종 {len(filtered_startups)}개 스타트업 저장 완료 → {INITIAL_STARTUP_CSV}")
        print("  --- 초기 목록 ---")
        for s in filtered_startups: print(f"  - {s}")
        print("  -----------------")
        return {"initial_candidates": filtered_startups}
    else:
        print(f"  ❌ AI 필터링 후 남은 스타트업이 없습니다.")
        pd.DataFrame(columns=["startup_name"]).to_csv(INITIAL_STARTUP_CSV, index=False, encoding="utf-8-sig")
        return {"initial_candidates": []}

def node_evaluate_all_and_rank(state: AgentState) -> Dict:
    """EvaluateAll & Rank: 전체 평가, 순위 매기기, CSV 저장"""
    print("\n[bold blue]=== ✨ 2단계: 전체 스타트업 상세 평가 및 순위 매기기 ===[/]")
    startup_list = state.get("initial_candidates", [])
    all_evaluations: List[Dict] = []

    if not startup_list:
        print("  ❌ 평가할 스타트업 목록이 없습니다.")
        return {"ranked_evaluations": [], "current_rank_index": 0}

    print(f"  ➡️ 총 {len(startup_list)}개 스타트업 평가 시작...")
    for i, startup_name in enumerate(startup_list):
        print(f"\n  ⭐ ({i+1}/{len(startup_list)}) 평가 대상: {startup_name}")
        startup_context = get_startup_context_for_eval(startup_name)
        evaluation_result = None
        if "실패" in startup_context or "없음" in startup_context:
            print("    ❌ 컨텍스트 부족/오류로 평가 불가")
            all_evaluations.append({"startup_name": startup_name, "error": "Context Retrieval Failed", "total_score": 0})
        else:
            evaluation_result = evaluate_startup_with_ai(startup_name, startup_context, EVALUATION_CRITERIA)
            if evaluation_result:
                total_score = sum(int(cat_data.get("score", 0)) for cat_data in evaluation_result.get("evaluation_summary", {}).values()) # int로 변환 보장
                evaluation_result["total_score"] = total_score
                all_evaluations.append(evaluation_result)
                print(f"    [bold yellow]✨ 총점: {total_score} / 30 ✨[/]")
            else:
                print("    ❌ 평가 결과 생성 실패")
                all_evaluations.append({"startup_name": startup_name, "error": "Evaluation Failed", "total_score": 0})
        # time.sleep(0.5) # 필요시 대기

    # 총점 기준 정렬
    sorted_evaluations = sorted(all_evaluations, key=lambda x: x.get("total_score", 0), reverse=True)

    # CSV 저장
    try:
        df_data = []
        for item in sorted_evaluations:
            row = {"startup_name": item.get("startup_name"),
                   "total_score": item.get("total_score"),
                   "overall_assessment": item.get("overall_assessment"),
                   "error": item.get("error")}
            if item.get("evaluation_summary"):
                for category, details in item["evaluation_summary"].items():
                    row[f"{category}_analysis"] = details.get("analysis")
                    row[f"{category}_score"] = details.get("score")
            df_data.append(row)
        df_ranked = pd.DataFrame(df_data)
        df_ranked.to_csv(RANKED_CSV_FILE, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ 총 {len(df_ranked)}개 스타트업 평가 결과 및 순위를 {RANKED_CSV_FILE}에 저장했습니다.")
    except Exception as e:
        print(f"\n  ❌ 평가 결과를 CSV 파일({RANKED_CSV_FILE})로 저장하는 중 오류 발생: {e}")

    return {"ranked_evaluations": sorted_evaluations, "current_rank_index": 0}

def node_select_ranked_startup(state: AgentState) -> Dict:
    """Select Ranked Startup: 순위 목록에서 다음 스타트업 선택"""
    print("\n[bold blue]=== 🎯 3단계: 순위 기반 스타트업 선택 ===[/]")
    ranked_list = state.get("ranked_evaluations", [])
    current_index = state.get("current_rank_index", 0)

    while current_index < len(ranked_list):
        selected_startup_details = ranked_list[current_index]
        # 평가 오류가 있었거나 점수가 0 이하인 스타트업은 건너뛰기
        if selected_startup_details.get("error") or selected_startup_details.get("total_score", 0) <= 0:
             print(f"  ⚠️ {current_index + 1}순위 '{selected_startup_details.get('startup_name')}' 건너뛰기 (오류 또는 0점). 다음 순위 시도...")
             current_index += 1 # 다음 인덱스로
        else:
            # 유효한 후보를 찾으면 반환
            print(f"  ✅ {current_index + 1}순위 선택: [bold green]{selected_startup_details.get('startup_name')}[/] (총점: {selected_startup_details.get('total_score')})")
            return {
                "current_startup_details": selected_startup_details,
                "current_rank_index": current_index + 1 # 다음 선택을 위해 인덱스 증가
            }

    # 루프를 다 돌았는데 유효한 후보가 없으면 종료
    print("  ❌ 더 이상 선택할 유효한 후보 순위가 없습니다.")
    return {"current_startup_details": None, "current_rank_index": current_index} # None 반환


# --- Agents B-H (검증 로직) ---
PROMPT_VALIDATE = ChatPromptTemplate.from_template(
    """System: 당신은 VC 투자 심사역입니다. 주어진 스타트업 정보와 평가 요약을 보고, '{criteria_check}' 기준을 충족하는지 **간단히 'pass' 또는 'fail'**로만 판단해주세요. 추가 설명 없이 오직 'pass' 또는 'fail' 단어만 출력해야 합니다.

Human: 스타트업 이름: {startup_name}
종합 평가 점수: {total_score} / 30
종합 의견: {overall_assessment}

검증 기준: {criteria_check}

판단 ('pass' 또는 'fail'):"""
)

def run_validation_agent(startup_details: Dict, criteria_check: str, agent_name: str) -> Literal["continue", "reject"]:
    """B-H 검증 에이전트 실행 함수"""
    print(f"--- 👤 Agent {agent_name}: 검증 시작 ---")
    if not startup_details:
        print(f"  ❌ 검증할 스타트업 정보 없음 -> REJECT")
        return "reject"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = PROMPT_VALIDATE.format_messages(
        startup_name=startup_details.get("startup_name", "N/A"),
        total_score=startup_details.get("total_score", "N/A"),
        overall_assessment=startup_details.get("overall_assessment", "N/A"),
        criteria_check=criteria_check
    )

    try:
        # LLM 호출 시도 (네트워크 오류 등 대비)
        for _ in range(3): # 최대 3번 재시도
            try:
                response = llm.invoke(prompt).content.strip().lower()
                if response in ["pass", "fail"]:
                    decision = "continue" if response == "pass" else "reject"
                    print(f"  ↳ 검증 결과: {decision.upper()}")
                    return decision
                else:
                    print(f"  ⚠️ Agent {agent_name} 예상치 못한 응답: '{response}' -> REJECT 처리")
                    return "reject"
            except Exception as retry_e:
                print(f"  ⚠️ Agent {agent_name} LLM 호출 재시도 중... 오류: {retry_e}")
                time.sleep(1) # 잠시 대기 후 재시도
        # 재시도 모두 실패
        print(f"  ❌ Agent {agent_name} LLM 호출 최종 실패 -> REJECT 처리")
        return "reject"
    except Exception as e:
        print(f"  ❌ Agent {agent_name} 검증 중 오류 발생: {e} -> REJECT 처리")
        return "reject"

def node_agent_b_validate(state: AgentState) -> Dict:
    decision = run_validation_agent(state.get("current_startup_details"),
                                     "초기 '고위험-고성장' 프로필 (목적, 속도, 아이디어, 불확실성, 자금, 목표)에 부합하는가?", "B")
    return {"last_decision": decision}

def node_agent_c_validate(state: AgentState) -> Dict:
    decision = run_validation_agent(state.get("current_startup_details"),
                                     "기술적 혁신성이나 경쟁 우위가 충분히 입증되었는가?", "C")
    return {"last_decision": decision}

def node_agent_d_validate(state: AgentState) -> Dict:
    decision = run_validation_agent(state.get("current_startup_details"),
                                     "실제 학습 효과나 사용자 만족도 근거가 제시되었는가?", "D")
    return {"last_decision": decision}

def node_agent_e_validate(state: AgentState) -> Dict:
    decision = run_validation_agent(state.get("current_startup_details"),
                                     "시장 규모, 수익 모델, 고객 확보 측면에서 매력적인가?", "E")
    return {"last_decision": decision}

def node_agent_f_validate(state: AgentState) -> Dict:
    decision = run_validation_agent(state.get("current_startup_details"),
                                     "경쟁사 대비 차별점이나 진입 장벽이 명확한가?", "F")
    return {"last_decision": decision}

def node_agent_g_validate(state: AgentState) -> Dict:
    decision = run_validation_agent(state.get("current_startup_details"),
                                     "시장 확장, 글로벌 진출 등 10배 이상 성장 잠재력이 보이는가?", "G")
    return {"last_decision": decision}

def node_agent_h_validate(state: AgentState) -> Dict:
    decision = run_validation_agent(state.get("current_startup_details"),
                                     "재무, 법률, 기술, 시장 등 주요 리스크가 관리 가능한 수준인가?", "H")
    return {"last_decision": decision}

def node_report_success(state: AgentState) -> Dict:
    """최종 통과 보고 노드"""
    print("\n" + "="*60)
    print(f"[bold green]🎉 최종 검증 통과 🎉[/]")
    startup = state.get('current_startup_details', {})
    rank = state.get('current_rank_index', 0) # 현재 인덱스는 다음 대상이므로 -1 하면 안됨
    print(f"  🏅 순위: {rank}")
    print(f"  ⭐ 스타트업: {startup.get('startup_name', '정보 없음')}")
    print(f"  ✨ 총점: {startup.get('total_score', 'N/A')} / 30")
    print(f"  📝 종합 의견: {startup.get('overall_assessment', 'N/A')}")
    if startup.get("evaluation_summary"):
        print("\n  --- 상세 점수 ---")
        for cat, details in startup["evaluation_summary"].items():
             print(f"    - {cat.capitalize()}: {details.get('score', 'N/A')}/5")
    print("="*60 + "\n")
    return {}

# === ⑥ LangGraph 그래프 구성 ===
def build_graph():
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("Agent_A_Generate_List", node_agent_a_generate_list)
    graph.add_node("Evaluate_All_And_Rank", node_evaluate_all_and_rank)
    graph.add_node("Select_Ranked_Startup", node_select_ranked_startup)
    graph.add_node("Agent_B_Validate", node_agent_b_validate)
    graph.add_node("Agent_C_Validate", node_agent_c_validate)
    graph.add_node("Agent_D_Validate", node_agent_d_validate)
    graph.add_node("Agent_E_Validate", node_agent_e_validate)
    graph.add_node("Agent_F_Validate", node_agent_f_validate)
    graph.add_node("Agent_G_Validate", node_agent_g_validate)
    graph.add_node("Agent_H_Validate", node_agent_h_validate)
    graph.add_node("Report_Success", node_report_success)

    # 진입점 설정
    graph.set_entry_point("Agent_A_Generate_List")

    # 엣지 및 라우터 정의
    def router_after_a(state: AgentState):
        return "evaluate_all" if state.get("initial_candidates") else END
    graph.add_conditional_edges("Agent_A_Generate_List", router_after_a, {
        "evaluate_all": "Evaluate_All_And_Rank", END: END
    })

    def router_after_evaluate(state: AgentState):
         return "select_ranked" if state.get("ranked_evaluations") else END
    graph.add_conditional_edges("Evaluate_All_And_Rank", router_after_evaluate, {
        "select_ranked": "Select_Ranked_Startup", END: END
    })

    def router_after_select(state: AgentState):
        # node_select_ranked_startup 에서 유효하지 않은 후보는 건너뛰고 None을 반환함
        return "validate_b" if state.get("current_startup_details") else END
    graph.add_conditional_edges("Select_Ranked_Startup", router_after_select, {
        "validate_b": "Agent_B_Validate", END: END # None이면 종료
    })

    # B-H 검증 라우터
    def router_after_validation(state: AgentState):
        return "proceed" if state.get("last_decision") == "continue" else "reject_and_select_new"

    graph.add_conditional_edges("Agent_B_Validate", router_after_validation, {
        "proceed": "Agent_C_Validate", "reject_and_select_new": "Select_Ranked_Startup" # 실패 시 다음 순위 선택
    })
    graph.add_conditional_edges("Agent_C_Validate", router_after_validation, {
        "proceed": "Agent_D_Validate", "reject_and_select_new": "Select_Ranked_Startup"
    })
    graph.add_conditional_edges("Agent_D_Validate", router_after_validation, {
        "proceed": "Agent_E_Validate", "reject_and_select_new": "Select_Ranked_Startup"
    })
    graph.add_conditional_edges("Agent_E_Validate", router_after_validation, {
        "proceed": "Agent_F_Validate", "reject_and_select_new": "Select_Ranked_Startup"
    })
    graph.add_conditional_edges("Agent_F_Validate", router_after_validation, {
        "proceed": "Agent_G_Validate", "reject_and_select_new": "Select_Ranked_Startup"
    })
    graph.add_conditional_edges("Agent_G_Validate", router_after_validation, {
        "proceed": "Agent_H_Validate", "reject_and_select_new": "Select_Ranked_Startup"
    })
    graph.add_conditional_edges("Agent_H_Validate", router_after_validation, {
        "proceed": "Report_Success", # 최종 통과
        "reject_and_select_new": "Select_Ranked_Startup" # 실패 시 다음 순위 시도
    })

    # 종료 엣지
    graph.add_edge("Report_Success", END)

    # 컴파일
    return graph.compile()

# === ⑦ 메인 실행 로직 ===
if __name__ == "__main__":
    app = build_graph()
    init_state: Dict = {}

    print("🚀 [bold]EdTech 투자 파이프라인 (A -> 평가/랭킹 -> 순위 기반 B-H 검증) 실행 시작[/]")

    try:
        # config에 recursion_limit 설정 (단계가 많으므로 넉넉하게 설정)
        final_state = app.invoke(init_state, config={"recursion_limit": 200}) # 예: 최대 200단계

        print("\n=== 그래프 실행 종료 ===")

        # 최종 상태 확인 (Report_Success 노드가 실행되었는지 여부 판단)
        # 마지막 노드가 Report_Success 이거나, last_decision이 continue 이고 current_startup_details가 있으면 성공
        # (더 정확한 방법은 LangSmith 추적을 보거나 Report_Success 노드에서 특정 flag 상태 추가)
        if final_state.get("last_decision") == "continue" and final_state.get("current_startup_details"):
             # 성공 메시지는 Report_Success 노드에서 이미 출력됨
             pass
        else:
            print("[bold red]❌ 최종 검증을 통과한 스타트업이 없습니다.[/]")


    except Exception as e:
        print(f"\n❌ 그래프 실행 중 치명적 오류 발생: {e}")
        traceback.print_exc()

    print("\n[bold magenta]=== 🎉 파이프라인 최종 완료 ===[/]")