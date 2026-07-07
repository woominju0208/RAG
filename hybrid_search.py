import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_core import vectorstore

_bm25_retriever: BM25Retriever | None = None
_company_index: dict[str, list[str]] | None = None

_CORP_SUFFIX_RE = re.compile(r"\(주\)|주식회사|\s+")


def _korean_bigram_tokenize(text: str) -> list[str]:
    grams = []
    for token in text.split():
        if len(token) < 2:
            grams.append(token)
        else:
            grams.extend(token[i:i + 2] for i in range(len(token) - 1))
    return grams


def _normalize_company(name: str) -> str:
    return _CORP_SUFFIX_RE.sub("", name)


def _get_company_index() -> dict[str, list[str]]:
    """정규화한 회사명 -> 실제 metadata에 저장된 표기 변형들. 같은 회사라도
    '(주)삼양패키징' / '삼양패키징' 처럼 파일마다 표기가 달라서 변형을 다 모아둔다."""
    global _company_index
    if _company_index is None:
        raw = vectorstore.get(include=["metadatas"])
        index: dict[str, list[str]] = {}
        for meta in raw["metadatas"]:
            company = meta.get("company")
            if not company:
                continue
            key = _normalize_company(company)
            variants = index.setdefault(key, [])
            if company not in variants:
                variants.append(company)
        _company_index = index
    return _company_index


def _detect_company_filter(question: str) -> dict | None:
    normalized_q = re.sub(r"\s+", "", question)
    matched_variants = []
    for key, variants in _get_company_index().items():
        if key and key in normalized_q:
            matched_variants.extend(variants)
    if not matched_variants:
        return None
    return {"company": {"$in": matched_variants}}


def _get_bm25_retriever(company_filter: dict | None) -> BM25Retriever | None:
    global _bm25_retriever
    if company_filter:
        raw = vectorstore.get(where=company_filter, include=["documents", "metadatas"])
        if not raw["documents"]:
            return None
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
        return BM25Retriever.from_documents(docs, preprocess_func=_korean_bigram_tokenize)

    if _bm25_retriever is None:
        raw = vectorstore.get(include=["documents", "metadatas"])
        if not raw["documents"]:
            return None
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
        _bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=_korean_bigram_tokenize)
    return _bm25_retriever


def _doc_key(doc: Document) -> tuple:
    return (doc.metadata.get("source"), doc.metadata.get("chunk_index"))


def hybrid_search(question: str, k: int = 5, pool: int = 15, rrf_k: int = 60) -> list[Document]:
    """벡터(의미) 검색과 BM25(키워드) 검색 결과를 RRF(Reciprocal Rank Fusion)로 합친다.
    질문에 저장된 회사명이 언급되면 그 회사의 청크로 먼저 범위를 좁힌 뒤 검색한다."""
    company_filter = _detect_company_filter(question)

    vector_docs = vectorstore.similarity_search(question, k=pool, filter=company_filter)

    bm25_retriever = _get_bm25_retriever(company_filter)
    if bm25_retriever is None:
        bm25_docs = []
    else:
        bm25_retriever.k = pool
        bm25_docs = bm25_retriever.invoke(question)

    scores: dict[tuple, float] = {}
    doc_by_key: dict[tuple, Document] = {}
    for ranked_docs in (vector_docs, bm25_docs):
        for rank, doc in enumerate(ranked_docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_by_key[key] = doc

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ranked_keys[:k]]
