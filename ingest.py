from pathlib import Path
from langchain_core.documents import Document
from dart_parser import parse_dart_xml
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
    return f"[{' | '.join(parts)}]\n" if parts else ""


def _store_chunks(path: Path, sections: list[Document]) -> None:
    chunks = text_splitter.split_documents(sections)
    if not chunks:
        print(f"{path.name}: 내용 없음, 건너뜀")
        return

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.page_content = _context_header(chunk.metadata) + chunk.page_content
    ids = [f"{path.stem}-{i}" for i in range(len(chunks))]

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


if __name__ == "__main__":
    main()
