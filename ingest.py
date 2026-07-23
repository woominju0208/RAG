from pathlib import Path
from langchain_core.documents import Document
from dart_parser import TABLE_ROW_SEP, parse_dart_xml
from hybrid_search import reset_cache
from rag_core import text_splitter, vectorstore

DATA_DIR = Path(__file__).resolve().parent / "data"


def _context_header(metadata: dict) -> str:
    """company/fiscal_period/section은 metadata로만 있으면 검색(임베딩·BM25)에 전혀
    영향을 못 준다 — 필터링과 최종 인용 라벨에만 쓰일 뿐이다. 표처럼 본문에 회사명이나
    연도가 안 나오는 청크는 그래서 "삼양패키징 매출실적" 같은 질문에서 밀리는 문제가 있었다.
    청크 텍스트 맨 앞에 이 정보를 그대로 붙여서 색인 대상 텍스트 자체에 포함시킨다."""
    parts = []
    if metadata.get("company"):
        parts.append(metadata["company"])
    if metadata.get("ceo_name"):
        parts.append(f"대표이사 {metadata['ceo_name']}")
    if metadata.get("fiscal_period"):
        parts.append(f"사업연도 {metadata['fiscal_period']}")
    if metadata.get("section"):
        parts.append(metadata["section"])
    header = f"[{' | '.join(parts)}]\n" if parts else ""

    # affiliates(계열회사 명단)는 회사명 등과 달리 길고(수십 개) 그 섹션에만 의미가 있어서,
    # 전체 청크가 아니라 "계열회사" 섹션에 속한 청크에만 붙인다. 이러면 "삼양패키징 계열회사
    # 목록 알려줘" 같은 질문에서, 요약 문장 청크가 상세 명단 청크보다 랭킹에서 이겨도
    # (짧고 키워드 밀도가 높아서 자주 그렇다) 명단 자체는 어차피 같이 딸려온다.
    if metadata.get("affiliates") and "계열회사" in metadata.get("section", ""):
        header += f"계열회사: {metadata['affiliates']}\n"

    return header


TABLE_WHOLE_CHUNK_LIMIT = 1500  # 이 크기 이하인 표는 셀/행을 자르지 않고 통째로 한 청크로 둔다


def _split_table_rows(text: str, chunk_size: int = 800) -> list[str]:
    """표를 TABLE_ROW_SEP(진짜 행 경계)로만 나누고, 셀 안에 원래 있던 줄바꿈은 그대로 둔다.
    표가 커서 여러 조각이 되면, 첫 번째 행을 "헤더"로 보고 각 조각 맨 위에 반복해서 붙여
    조각 하나만 봐도 어느 컬럼인지 알 수 있게 한다. 다만 첫 행 자체가 아주 길면(전형적인
    짧은 헤더가 아니라 그 자체로 긴 내용이면) 반복하지 않는다 — 안 그러면 헤더만으로
    chunk_size를 넘겨버릴 수 있다."""
    rows = text.split(TABLE_ROW_SEP)
    if len(rows) <= 1:
        return ["\n".join(rows)]

    header = rows[0] if len(rows[0]) < 300 else None
    data_rows = rows[1:] if header is not None else rows
    header_len = len(header) + 1 if header else 0

    chunks = []
    current = [header] if header else []
    current_len = header_len
    for row in data_rows:
        row_len = len(row) + 1
        if current and current_len + row_len > chunk_size and len(current) > (1 if header else 0):
            chunks.append("\n".join(current))
            current = [header] if header else []
            current_len = header_len
        current.append(row)
        current_len += row_len
    if current:
        chunks.append("\n".join(current))
    return chunks or ["\n".join(rows)]


def _split_chunks(sections: list[Document]) -> list[Document]:
    """표는 크기에 따라: 작으면 통째로, 크면 행 단위로만 나눈다. 문단은 지금처럼
    RecursiveCharacterTextSplitter로 글자 수 기준으로 나눈다."""
    chunks = []
    for doc in sections:
        if not doc.metadata.get("is_table"):
            chunks.extend(text_splitter.split_documents([doc]))
        elif len(doc.page_content) > TABLE_WHOLE_CHUNK_LIMIT:
            for piece in _split_table_rows(doc.page_content):
                chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))
        else:
            whole = doc.page_content.replace(TABLE_ROW_SEP, "\n")
            chunks.append(Document(page_content=whole, metadata=dict(doc.metadata)))
    return chunks


def _store_chunks(path: Path, sections: list[Document]) -> None:
    chunks = _split_chunks(sections)
    if not chunks:
        print(f"{path.name}: 내용 없음, 건너뜀")
        return

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.page_content = _context_header(chunk.metadata) + chunk.page_content
    ids = [f"{path.stem}-{i}" for i in range(len(chunks))]

    for chunk in chunks:
        meta = chunk.metadata
        tag = "표" if meta.get("is_table") else "문단"
        preview = chunk.page_content.replace("\n", " ")[:60]
        print(f"  [청크 {meta['chunk_index']}] ({tag}, len={len(chunk.page_content)}) {meta.get('section')} | {preview}...")

    vectorstore.add_documents(chunks, ids=ids)
    print(f"{path.name}: {len(chunks)}개 청크 저장 완료")


def ingest_txt(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    _store_chunks(path, [Document(page_content=text, metadata={"source": path.name})])


def ingest_dart_xml(path: Path) -> None:
    _store_chunks(path, parse_dart_xml(path))


def main() -> None:
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    xml_files = sorted(DATA_DIR.rglob("*.xml"))

    if not txt_files and not xml_files:
        print(f"{DATA_DIR}에 .txt/.xml 파일이 없습니다. 문서를 넣고 다시 실행하세요.")
        return

    for path in txt_files:
        ingest_txt(path)

    for path in xml_files:
        ingest_dart_xml(path)

    reset_cache()


if __name__ == "__main__":
    main()
