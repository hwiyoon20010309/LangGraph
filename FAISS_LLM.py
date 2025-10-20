import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 🔹 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# ✅ OpenAI 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 OPENAI_API_KEY가 .env에 없습니다.")

# 🔹 임베딩 및 벡터 스토어 로드
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY   # ✅ 명시적으로 전달
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 🔹 검색기 + LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🔹 테스트 질의
query = "Squirrel AI의 기술력과 교육 성과를 요약해줘"
print(qa.run(query))
