import sys
from rag_core import embed, collection, client, CHAT_MODEL


def ask(question: str, top_k: int = 5) -> str:
    query_embedding = embed([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return "저장된 문서가 없습니다. 먼저 ingest.py를 실행해 문서를 넣어주세요."

    context = "\n\n".join(
        f"[{meta['source']} #{meta['chunk_index']}]\n{doc}"
        for doc, meta in zip(documents, metadatas)
    )

    prompt = (
        "아래 문서 발췌를 참고해서 질문에 답하세요. 문서에 근거가 없으면 모른다고 답하세요.\n\n"
        f"문서:\n{context}\n\n질문: {question}"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def main() -> None:
    if len(sys.argv) < 2:
        print('사용법: python query.py "질문 내용"')
        return

    question = " ".join(sys.argv[1:])
    print(ask(question))


if __name__ == "__main__":
    main()
