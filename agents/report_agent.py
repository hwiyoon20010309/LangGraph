"""
agents/report_agent.py
보고서 생성 Agent (PDF 출력 - 선택사항)
"""
import os
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from .base import AgentState, llm

# PDF 생성 라이브러리 (선택적 import)
PDF_AVAILABLE = False
try:
    import markdown
    from weasyprint import HTML, CSS
    PDF_AVAILABLE = True
except (ImportError, OSError) as e:
    # Windows GTK 문제 등으로 import 실패 시 무시
    pass


def markdown_to_pdf_weasyprint(markdown_text: str, output_path: str, startup_name: str) -> bool:
    """
    WeasyPrint를 사용한 PDF 변환
    """
    if not PDF_AVAILABLE:
        return False
    
    try:
        # 1. Markdown을 HTML로 변환
        html_content = markdown.markdown(
            markdown_text,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
        
        # 2. HTML 템플릿 생성 (한글 폰트 지원)
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{startup_name} 투자 심사 보고서</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Noto Sans KR', 'Malgun Gothic', '맑은 고딕', sans-serif;
            line-height: 1.8;
            color: #333;
            padding: 40px;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #ffffff;
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            padding-bottom: 15px;
            border-bottom: 4px solid #3498db;
        }}
        
        h2 {{
            color: #34495e;
            font-size: 24px;
            font-weight: 600;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        h3 {{
            color: #7f8c8d;
            font-size: 18px;
            font-weight: 500;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        strong {{
            color: #2980b9;
            font-weight: 600;
        }}
        
        ul, ol {{
            margin-left: 30px;
            margin-bottom: 15px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        
        .header h1 {{
            color: white;
            border-bottom: none;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            font-size: 12px;
            color: #95a5a6;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 투자 심사 보고서</h1>
        <p>{startup_name} | {datetime.now().strftime('%Y년 %m월 %d일')}</p>
    </div>
    
    {html_content}
    
    <div class="footer">
        <p>본 보고서는 AI 기반 자동 분석 시스템에 의해 생성되었습니다.</p>
        <p>© {datetime.now().year} Investment Analysis System. All rights reserved.</p>
    </div>
</body>
</html>
"""
        
        # 3. HTML을 PDF로 변환
        HTML(string=html_template).write_pdf(output_path)
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ WeasyPrint PDF 생성 실패: {e}")
        return False


def save_html_report(markdown_text: str, output_path: str, startup_name: str) -> bool:
    """
    HTML 보고서 저장 (PDF 대체 방법)
    브라우저에서 열어서 PDF로 출력 가능
    """
    try:
        import markdown
        
        html_content = markdown.markdown(
            markdown_text,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
        
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{startup_name} 투자 심사 보고서</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Noto Sans KR', 'Malgun Gothic', '맑은 고딕', sans-serif;
            line-height: 1.8;
            color: #333;
            padding: 40px;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #ffffff;
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            padding-bottom: 15px;
            border-bottom: 4px solid #3498db;
        }}
        
        h2 {{
            color: #34495e;
            font-size: 24px;
            font-weight: 600;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        h3 {{
            color: #7f8c8d;
            font-size: 18px;
            font-weight: 500;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        strong {{
            color: #2980b9;
            font-weight: 600;
        }}
        
        ul, ol {{
            margin-left: 30px;
            margin-bottom: 15px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        
        .header h1 {{
            color: white;
            border-bottom: none;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            font-size: 12px;
            color: #95a5a6;
        }}
        
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .print-button:hover {{
            background: #2980b9;
        }}
        
        @media print {{
            .print-button {{
                display: none;
            }}
            
            body {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-button" onclick="window.print()">🖨️ PDF로 저장</button>
    
    <div class="header">
        <h1>📊 투자 심사 보고서</h1>
        <p>{startup_name} | {datetime.now().strftime('%Y년 %m월 %d일')}</p>
    </div>
    
    {html_content}
    
    <div class="footer">
        <p>본 보고서는 AI 기반 자동 분석 시스템에 의해 생성되었습니다.</p>
        <p>© {datetime.now().year} Investment Analysis System. All rights reserved.</p>
        <p style="margin-top: 10px; font-size: 11px;">
            💡 브라우저에서 Ctrl+P (또는 우측 상단 버튼)를 눌러 PDF로 저장할 수 있습니다.
        </p>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ HTML 저장 실패: {e}")
        return False


def report_generation_agent(state: AgentState) -> AgentState:
    """Agent 8: 최종 보고서 생성 - Markdown + HTML (PDF 선택)"""
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
    
    markdown_content = response.content
    
    # 파일 저장
    output_dir = "investment_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"{state['startup_name']}_투자분석_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Markdown 파일 저장 (항상)
    md_filepath = os.path.join(output_dir, f"{base_filename}.md")
    with open(md_filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"   ✅ Markdown 저장: {md_filepath}")
    
    # 2. HTML 파일 저장 (항상 - 브라우저에서 PDF 출력 가능)
    html_filepath = os.path.join(output_dir, f"{base_filename}.html")
    if save_html_report(markdown_content, html_filepath, state["startup_name"]):
        print(f"   ✅ HTML 저장: {html_filepath}")
        print(f"      💡 브라우저로 열어서 Ctrl+P로 PDF 저장 가능")
        final_path = html_filepath
    else:
        final_path = md_filepath
    
    # 3. PDF 파일 생성 시도 (WeasyPrint 사용 가능한 경우만)
    if PDF_AVAILABLE:
        pdf_filepath = os.path.join(output_dir, f"{base_filename}.pdf")
        if markdown_to_pdf_weasyprint(markdown_content, pdf_filepath, state["startup_name"]):
            print(f"   ✅ PDF 저장: {pdf_filepath}")
            final_path = pdf_filepath
        else:
            print(f"   ⚠️ WeasyPrint PDF 생성 실패")
    else:
        print(f"   ℹ️ WeasyPrint 미설치 - HTML 파일을 브라우저로 열어 PDF 저장 가능")
    
    print(f"✅ [Agent 8] 완료 - 보고서 저장: {final_path}")
    
    # 자신의 필드만 반환
    return {
        "report": markdown_content,
        "pdf_path": final_path
    }