import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).resolve().parent / ".env")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

api_key = os.getenv("OPENAI_API_KEY", "").strip()

embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)
llm = ChatOpenAI(model=CHAT_MODEL, api_key=api_key)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# chromadb의 HNSW 인덱스가 경로에 비ASCII 문자(예: 한글 폴더명)가 있으면 깨지는
# 문제가 있어(Rust 코어 버그), 벡터 인덱스는 프로젝트 폴더 밖 영문 경로에 저장한다.
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path.home() / ".rag_project_chroma"))

vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
