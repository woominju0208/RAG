import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_core import vectorstore

_bm25_retriever: BM25Retriever | None = None
_company_index: dict[str, list[str]] | None = None


def reset_cache() -> None:
    """ingest.py가 새 문서를 추가한 뒤 반드시 호출해야 한다. _company_index/_bm25_retriever는
    한 번 빌드되면 재사용하는 모듈 전역 캐시라, 이걸 안 비우면 같은 프로세스(노트북 커널 등)
    안에서는 새로 들어온 회사가 회사명 필터에 잡히지 않아 필터링이 조용히 깨진다."""
    global _bm25_retriever, _company_index
    _bm25_retriever = None
    _company_index = None

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


def _match_company_keys(question: str) -> list[str]:
    normalized_q = re.sub(r"\s+", "", question)
    return [key for key in _get_company_index() if key and key in normalized_q]


def _detect_company_filter(question: str) -> dict | None:
    keys = _match_company_keys(question)
    if not keys:
        return None
    matched_variants = []
    for key in keys:
        matched_variants.extend(_get_company_index()[key])
    return {"company": {"$in": matched_variants}}


def _strip_company_mentions(question: str) -> str:
    """회사 필터를 이미 적용했으면, 검색 문장에 회사명이 남아있을 때 BM25/벡터 점수가
    회사명 반복 횟수에 좌우되어 정작 찾는 키워드(예: 매출실적)가 밀리는 문제가 있다.
    필터링된 범위 안에서는 회사명이 더 이상 구분 신호가 아니므로 검색 문장에서 뺀다."""
    keys = _match_company_keys(question)
    stripped = question
    for key in keys:
        stripped = stripped.replace(key, " ")
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or question


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


def hybrid_search(question: str, k: int = 8, pool: int = 15, rrf_k: int = 60) -> list[Document]:
    """벡터(의미) 검색과 BM25(키워드) 검색 결과를 RRF(Reciprocal Rank Fusion)로 합친다.
    질문에 저장된 회사명이 언급되면 그 회사의 청크로 먼저 범위를 좁힌 뒤 검색한다.

    회사 필터가 걸리면 검색어를 회사명 포함/제외 두 버전으로 만들어 둘 다 검색한다.
    회사명을 빼면 "매출실적"처럼 회사명이 반복 언급되는 청크에 밀리던 키워드가 잘 잡히지만,
    남는 검색어가 "대표이사"처럼 너무 짧고 흔한 단어면 오히려 회사명이 실제로 적힌 정답 청크
    (예: 재무제표 서명란의 "삼양애니팜 대표이사 민필홍")보다 회사명 없이 그 단어만 반복되는
    엉뚱한 청크가 더 유사하다고 나오는 역효과가 있다. 두 버전을 다 검색해서 합치면 어느 한쪽만
    잡아내는 경우도 살아남는다."""
    company_filter = _detect_company_filter(question)
    search_texts = [question]
    if company_filter:
        stripped = _strip_company_mentions(question)
        if stripped != question:
            search_texts.append(stripped)

    bm25_retriever = _get_bm25_retriever(company_filter)
    if bm25_retriever is not None:
        bm25_retriever.k = pool

    ranked_lists = []
    for text in search_texts:
        ranked_lists.append(vectorstore.similarity_search(text, k=pool, filter=company_filter))
        if bm25_retriever is not None:
            ranked_lists.append(bm25_retriever.invoke(text))

    scores: dict[tuple, float] = {}
    doc_by_key: dict[tuple, Document] = {}
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_by_key[key] = doc

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    top_keys = ranked_keys[:k]

    filter_desc = ", ".join(company_filter["company"]["$in"]) if company_filter else "없음"
    print(f"[검색] '{question}' | 회사 필터: {filter_desc}")
    for rank, key in enumerate(top_keys, start=1):
        doc = doc_by_key[key]
        source, chunk_index = key
        print(f"  {rank}위 {source} #{chunk_index} (score {scores[key]:.4f}) | {doc.metadata.get('section')}")

    return [doc_by_key[key] for key in top_keys]
