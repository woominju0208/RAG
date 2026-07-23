import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from hybrid_search import hybrid_search, _detect_company_filter
from rag_core import llm

PROMPT = ChatPromptTemplate.from_template(
    "아래 문서 발췌를 참고해서 질문에 답하세요. 문서에 근거가 없으면 모른다고 답하세요.\n\n"
    "문서:\n{context}\n\n질문: {question}"
)


def format_docs(docs) -> str:
    # company/fiscal_period/section은 ingest.py가 청크 본문 맨 앞에 이미 붙여뒀으므로
    # (예: "[삼양패키징 | 사업연도 2020.01.01~2020.06.30 | II. 사업의 내용]"),
    # 여기서는 어느 파일의 몇 번째 청크인지만 표시해 중복을 피한다.
    parts = [f"[{doc.metadata['source']} #{doc.metadata['chunk_index']}]\n{doc.page_content}" for doc in docs]
    return "\n\n".join(parts)


def ask(question: str, top_k: int = 8) -> str:
    docs = hybrid_search(question, k=top_k)

    if not docs:
        return "저장된 문서가 없습니다. 먼저 ingest.py를 실행해 문서를 넣어주세요."

    if _detect_company_filter(question) is None:
        companies = sorted({doc.metadata.get("company") for doc in docs if doc.metadata.get("company")})
        if len(companies) > 1:
            return (
                "질문에 회사명이 없어서 여러 회사의 문서가 섞여 검색됐습니다: "
                f"{', '.join(companies)}. 어느 회사인지 질문에 포함해서 다시 물어봐 주세요."
            )

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
