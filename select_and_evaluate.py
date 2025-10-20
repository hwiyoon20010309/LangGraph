import os
import re
import requests
import pandas as pd
import random
import json
import time # [💡] 대기 시간 사용 위해 추가
from typing import List, Set, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from rich import print

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
STARTUP_CSV_FILE = "ai_filtered_startups.csv"
# [💡💡💡] 순위 결과를 저장할 새 CSV 파일명
RANKED_CSV_FILE = "ranked_startup_evaluations.csv"

# === ②-1. 스타트업 목록 생성을 위한 검색 쿼리 ===
AGGREGATION_QUERIES = [
    "AI 교육 스타트업 목록 (예: Riiid, Mathpresso, Sana Labs, Squirrel AI 등)",
    "EdTech 분야에서 인공지능(AI)을 활용하는 주요 스타트업",
    "Top AI education or EdTech startups 2025 funding news",
    "AI tutoring platform startups using adaptive learning investment",
]

# === ②-2. 스타트업 목록 생성을 위한 정규식 + 필터 ===
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

# === ②-3. 스타트업 평가 기준 (비즈니스 성장 중심) ===
EVALUATION_CRITERIA = {
    "purpose": [  # 목적: 빠른 확장과 시장 선점
        "회사의 주된 목표가 시장 점유율의 빠른 확보인가?",
        "단기적인 수익성보다 장기적인 시장 지배력을 우선하는가?",
        "공격적인 성장 전략(예: 대규모 마케팅, 빠른 제품 출시)을 추구하는가?",
        "기존 시장을 파괴(disrupt)하려는 명확한 비전이 있는가?",
        "네트워크 효과나 규모의 경제를 통해 시장을 선점하려는 계획이 있는가?"
    ],
    "growth_speed": [  # 성장 속도: 상당히 빠름 (10배 성장 등)
        "사용자, 매출 등 핵심 지표의 성장 목표가 매우 높은가? (예: 연 10배)",
        "과거 성장률이 업계 평균을 크게 상회했는가?",
        "단기간 내 폭발적인 성장을 달성할 수 있는 잠재력이 있는가?",
        "성장 속도를 가속화하기 위한 구체적인 계획(예: 인재 영입, 기술 투자)이 있는가?",
        "시장 변화에 빠르게 적응하며 성장 모멘텀을 유지할 수 있는가?"
    ],
    "idea": [  # 아이디어: 혁신적이고 독창적인 기술, 서비스 중심
        "핵심 기술이나 서비스가 기존 방식과 비교해 명확히 혁신적인가?",
        "독자적인 기술(예: 특허)이나 차별화된 비즈니스 모델을 가지고 있는가?",
        "아이디어가 모방하기 어렵거나 상당한 진입 장벽을 가지고 있는가?",
        "기술/서비스가 잠재적으로 새로운 시장을 창출할 수 있는가?",
        "아이디어가 명확하고 설득력 있게 전달되는가?"
    ],
    "uncertainty": [  # 불확실성: 매우 높음 (시장, 기술, 고객 관점)
        "타겟 시장의 반응이 아직 검증되지 않았는가?",
        "핵심 기술이 상용화 초기 단계이거나 아직 개발 중인가?",
        "고객의 행동이나 니즈 변화에 대한 예측이 어려운가?",
        "경쟁 환경이 빠르게 변하거나 예측 불가능한 요소가 많은가?",
        "사업 모델이나 수익화 방식에 대한 불확실성이 존재하는가?"
    ],
    "funding": [  # 자금 조달: 투자 중심 (VC, 엔젤 등)
        "주요 자금 조달 방식이 외부 투자 유치(VC, 엔젤 등)인가?",
        "과거에 상당 규모의 투자를 유치한 이력이 있는가?",
        "향후 대규모 투자 유치를 계획하고 있는가?",
        "투자자들이 매력적으로 느낄 만한 성장 스토리를 가지고 있는가?",
        "매출이나 자체 현금 흐름보다 투자금에 의존하여 운영되는 경향이 있는가?"
    ],
    "final_goal": [  # 최종 목표: M&A, IPO
        "회사의 장기적인 목표가 M&A(인수합병)인가?",
        "회사의 장기적인 목표가 IPO(기업공개)인가?",
        "창업자나 경영진이 명확한 Exit 전략을 가지고 있는가?",
        "M&A나 IPO를 가능하게 할 만한 규모나 시장 지위를 목표로 하는가?",
        "투자자들이 Exit을 기대할 만한 구조를 가지고 있는가?"
    ]
}


# === ③-1. Tavily 검색 함수 (목록 생성용) ===
def tavily_search_for_aggregation(query: str, max_results: int = 40) -> dict:
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    payload = { "query": query, "max_results": min(max_results, 50), "include_answer": True, "search_depth": "advanced" }
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

# === ③-2. 이름 후보 추출 함수 ===
def extract_candidate_names(*texts: str) -> Set[str]:
    candidates: Set[str] = set()
    for text in texts:
        if not text: continue
        for match in NAME_PATTERN.findall(text):
            name = match.strip()
            if (len(name) < 3 or name.upper() in STOPWORDS or
                not any(c.islower() for c in name if c.isalpha()) or
                len(name.split()) > 3 or
                not re.match(r"^[A-Z]", name) or
                re.search(r'\d{4}', name)):
                continue
            candidates.add(name)
    return candidates

# === ③-3. AI 필터 함수 (GPT로 실제 기업명만 남김) ===
def ai_filter_startups(candidates: List[str]) -> List[str]:
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

# === ③-4. 전체 스타트업 목록 생성 및 저장 함수 ===
def generate_and_save_startup_list(output_csv: str) -> List[str]:
    print("[bold blue]=== 🚀 1단계: AI EdTech 스타트업 목록 생성 시작 ===[/]")
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
         print("  ❌ 1차 추출된 후보가 없습니다. 쿼리를 확인하세요.")
         return []

    print("  🤖 AI 필터링 진행 중...")
    filtered_startups = ai_filter_startups(sorted(list(all_candidates)))

    if filtered_startups:
        df = pd.DataFrame(filtered_startups, columns=["startup_name"])
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"  ✅ 최종 {len(filtered_startups)}개 스타트업 저장 완료 → {output_csv}")
        print("  --- 최종 목록 ---")
        for s in filtered_startups: print(f"  - {s}")
        print("  -----------------")
        return filtered_startups
    else:
        print(f"  ❌ AI 필터링 후 남은 스타트업이 없습니다.")
        pd.DataFrame(columns=["startup_name"]).to_csv(output_csv, index=False, encoding="utf-8-sig")
        return []

# === ④-1. Tavily로 상세 평가용 컨텍스트 검색 함수 ===
def get_startup_context_for_eval(startup_name: str, max_results: int = 7) -> str:
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
        return full_context[:15000]

    except requests.exceptions.RequestException as e:
        print(f"  ❌ (상세 검색) Tavily 네트워크 오류: {e}")
        return f"Tavily 상세 검색 실패: {e}"
    except Exception as e:
        print(f"  ❌ (상세 검색) Tavily 처리 중 오류: {e}")
        return f"Tavily 상세 검색 실패: {e}"

# === ④-2. AI 상세 평가 함수 ===
def evaluate_startup_with_ai(startup_name: str, context: str, criteria: Dict[str, List[str]]) -> Optional[Dict]:
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

# === ⑤ 메인 실행 로직 (💡💡💡 순차 평가 -> 순위 저장 로직 추가 💡💡💡) ===
if __name__ == "__main__":
    # 1단계: 스타트업 목록 생성 및 저장
    startup_list = generate_and_save_startup_list(STARTUP_CSV_FILE)

    print("\n" + "="*60 + "\n")

    # 2단계: 목록의 모든 스타트업 순차 평가 및 결과 저장
    all_evaluations: List[Dict] = [] # 모든 평가 결과를 저장할 리스트

    if not startup_list:
        print("[bold red]➡️ 생성된 스타트업 목록이 없어 상세 평가를 진행할 수 없습니다.[/]")
    else:
        print(f"[bold blue]=== ✨ 2단계: 총 {len(startup_list)}개 스타트업 순차 상세 평가 시작 ===[/]")

        for i, selected_startup in enumerate(startup_list):
            print(f"\n[bold sky_blue1]⭐ ({i+1}/{len(startup_list)}) 평가 대상: {selected_startup} ⭐[/]")

            startup_context = get_startup_context_for_eval(selected_startup)

            evaluation_result = None # 평가 결과 초기화
            if "실패" in startup_context or "없음" in startup_context:
                print("  ❌ 컨텍스트 검색 실패 또는 정보 부족으로 평가를 건너<0xEB><0x9B><0x81>니다.")
                all_evaluations.append({"startup_name": selected_startup, "error": "Context Retrieval Failed", "total_score": 0})
            else:
                evaluation_result = evaluate_startup_with_ai(
                    selected_startup, startup_context, EVALUATION_CRITERIA
                )

                print("\n" + "-"*50)
                print(f"  [bold green]📊 ({i+1}/{len(startup_list)}) 평가 결과: {selected_startup} 📊[/]")
                if evaluation_result:
                    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))

                    total_score = 0
                    if evaluation_result.get("evaluation_summary"):
                        for category_data in evaluation_result["evaluation_summary"].values():
                            total_score += category_data.get("score", 0)
                    evaluation_result["total_score"] = total_score
                    all_evaluations.append(evaluation_result)
                    print(f"  [bold yellow]✨ 총점: {total_score} / 30 ✨[/]")

                else:
                    print("  ❌ 평가 결과를 생성하지 못했습니다.")
                    all_evaluations.append({"startup_name": selected_startup, "error": "Evaluation Failed", "total_score": 0})
                print("-"*50)

            # API 호출 속도 조절 (예: 0.5초 대기)
            # print("  ⏳ 다음 평가까지 0.5초 대기...")
            # time.sleep(0.5)

    print("\n[bold magenta]=== 🏁 모든 스타트업 평가 완료 ===[/]")

    # 3단계: 평가 결과 정렬 및 CSV 저장 (이름과 점수만) + 최고점 스타트업 발표
    print("\n" + "="*60 + "\n")
    print(f"[bold blue]=== 💾 3단계: 평가 결과 정렬 및 {RANKED_CSV_FILE} 저장 (이름, 점수) ===[/]")

    if not all_evaluations:
        print("❌ 평가된 스타트업이 없어 결과를 저장할 수 없습니다.")
    else:
        # 총점(total_score) 기준으로 내림차순 정렬
        sorted_evaluations = sorted(all_evaluations, key=lambda x: x.get("total_score", 0), reverse=True)

        # [💡💡💡] 이름과 총점만 추출하여 DataFrame 생성
        ranked_data = [
            {"startup_name": item.get("startup_name"), "total_score": item.get("total_score", 0)}
            for item in sorted_evaluations if "error" not in item # 오류가 없는 결과만 포함
        ]

        if not ranked_data:
             print(f"❌ 유효한 평가 결과가 없어 {RANKED_CSV_FILE} 파일을 생성하지 않습니다.")
        else:
            try:
                df_ranked = pd.DataFrame(ranked_data)
                # CSV 파일로 저장 (이름, 점수만)
                df_ranked.to_csv(RANKED_CSV_FILE, index=False, encoding="utf-8-sig")
                print(f"✅ 총 {len(df_ranked)}개 스타트업의 이름과 점수를 순위대로 {RANKED_CSV_FILE}에 저장했습니다.")

                # [💡💡💡] 최고점 스타트업 발표 (정렬된 목록의 첫 번째 항목만)
                top_startup = df_ranked.iloc[0]
                top_score = top_startup["total_score"]

                # 동점자가 있는지 확인 (정보 제공 목적)
                top_startups_df = df_ranked[df_ranked["total_score"] == top_score]

                print(f"\n🏅 최고점 스타트업 (다음 단계 진행 대상): [bold yellow]{top_startup['startup_name']}[/] (총점: {int(top_score)} / 30)") # int로 변환하여 소수점 제거

                if len(top_startups_df) > 1:
                    print(f"   (참고: 총 {len(top_startups_df)}개의 스타트업이 최고점 동점입니다. {RANKED_CSV_FILE} 참조)")

            except Exception as e:
                print(f"❌ 평가 결과를 CSV 파일로 저장하는 중 오류 발생: {e}")

    print("\n[bold magenta]=== 🎉 파이프라인 최종 완료 ===[/]")