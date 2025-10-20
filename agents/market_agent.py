"""
agents/market_agent.py
시장성 분석 Agent (RAG 포함)
"""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
try:
    from langchain_tavily import TavilySearch
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults as TavilySearch
from .base import AgentState, EVALUATION_CRITERIA, llm, extract_score


def market_agent(state: AgentState) -> AgentState:
    """Agent 3: 시장성 분석 (RAG 포함)"""
    print("\n💰 [Agent 3] 시장성 분석 시작...")
    
    startup_name = state["startup_name"]
    checklist = EVALUATION_CRITERIA["market"]
    
    # PDF RAG
    pdf_path = os.getenv("PDF_PATH", "")
    rag_context = ""
    if pdf_path and os.path.exists(pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
            vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved = retriever.get_relevant_documents(f"{startup_name} 교육 시장")
            rag_context = "\n".join([doc.page_content for doc in retrieved])
        except:
            rag_context = "PDF 로딩 실패"
    
    # Web Search
    try:
        search = TavilySearch(max_results=10)
        results = search.invoke(f"{startup_name} 교육 시장 규모")
        # TavilySearch는 문자열을 직접 반환하거나 리스트를 반환할 수 있음
        if isinstance(results, str):
            web_context = results
        elif isinstance(results, list):
            web_context = "\n".join([f"- {r.get('title', r.get('content', str(r)))}" for r in results])
        else:
            web_context = str(results)
    except Exception as e:
        web_context = f"검색 실패: {str(e)}"
    
    combined = f"[PDF 자료]\n{rag_context}\n\n[웹 검색]\n{web_context}"
    
    prompt = ChatPromptTemplate.from_template("""
교육 스타트업 '{startup_name}'의 시장성을 평가하세요.

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
        "context": combined
    })
    
    analysis = response.content
    score = extract_score(analysis)
    
    print(f"✅ [Agent 3] 완료 - 시장성 점수: {score}")
    
    # 자신의 필드만 반환
    return {
        "market_score": score,
        "market_analysis_evidence": analysis
    }