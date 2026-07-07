from pathlib import Path
from langchain_core.documents import Document
from dart_parser import parse_dart_xml
from rag_core import text_splitter, vectorstore

DATA_DIR = Path(__file__).resolve().parent / "data"


def _store_chunks(path: Path, sections: list[Document]) -> None:
    chunks = text_splitter.split_documents(sections)
    if not chunks:
        print(f"{path.name}: 내용 없음, 건너뜀")
        return

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
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
