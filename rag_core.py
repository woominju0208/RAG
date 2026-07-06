import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())

chroma_client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent / ".chroma"))
collection = chroma_client.get_or_create_collection("documents")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]
