import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from hybrid_search import hybrid_search
from rag_core import llm

PROMPT = ChatPromptTemplate.from_template(
    "아래 문서 발췌를 참고해서 질문에 답하세요. 문서에 근거가 없으면 모른다고 답하세요.\n\n"
    "문서:\n{context}\n\n질문: {question}"
)


def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        meta = doc.metadata
        label = f"{meta['source']} #{meta['chunk_index']}"
        if meta.get("section"):
            label += f" | {meta['section']}"
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ask(question: str, top_k: int = 5) -> str:
    docs = hybrid_search(question, k=top_k)

    if not docs:
        return "저장된 문서가 없습니다. 먼저 ingest.py를 실행해 문서를 넣어주세요."

    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})


def main() -> None:
    if len(sys.argv) < 2:
        print('사용법: python query.py "질문 내용"')
        return

    question = " ".join(sys.argv[1:])
    print(ask(question))


if __name__ == "__main__":
    main()
