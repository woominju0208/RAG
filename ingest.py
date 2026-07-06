from pathlib import Path
from rag_core import chunk_text, embed, collection

DATA_DIR = Path(__file__).resolve().parent / "data"


def ingest_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    if not chunks:
        print(f"{path.name}: 내용 없음, 건너뜀")
        return

    embeddings = embed(chunks)
    ids = [f"{path.stem}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"{path.name}: {len(chunks)}개 청크 저장 완료")


def main() -> None:
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        print(f"{DATA_DIR}에 .txt 파일이 없습니다. 문서를 넣고 다시 실행하세요.")
        return

    for path in files:
        ingest_file(path)


if __name__ == "__main__":
    main()
