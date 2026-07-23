"""RAG 정확도 평가 스크립트.

python eval.py 로 실행하면 EVAL_CASES를 순회하며

- 검색(Retrieval) 지표: 질문에 해당하는 회사의 청크가 hybrid_search 상위 k개 안에
  들어왔는지(Hit Rate@k), 몇 위로 들어왔는지(MRR, Mean Reciprocal Rank)
- 답변(Answer) 지표: query.ask()의 최종 답변에 정답 키워드가 들어있는지(정확도)

를 계산해서 표로 출력한다.

CEO/계열회사 질문의 정답 키워드는 하드코딩하지 않고, 지금 vectorstore에 저장된
ceo_name/affiliates metadata에서 직접 가져온다 — 이렇게 해야 재수집으로 데이터가
바뀌어도 eval 스크립트를 따로 고칠 필요가 없다. 반면 매출실적/유동자산처럼 구조화된
metadata 필드가 없는 숫자 질문은 "모른다류 응답이 아니면 일단 통과"로 보는 약한 검증만
하며(expected_keywords를 채우면 정확 매칭으로 강화 가능), 이 한계는 결과 표에 표시한다.
"""

import sys

from hybrid_search import hybrid_search, _normalize_company, _get_company_index
from query import ask
from rag_core import vectorstore

sys.stdout.reconfigure(encoding="utf-8")

TOP_K = 8

EVAL_CASES = [
    {"question": "삼양애니팜 대표이사가 누구야?", "type": "ceo", "company": "삼양애니팜"},
    {"question": "삼양패키징 대표이사가 누구야?", "type": "ceo", "company": "삼양패키징"},
    {"question": "삼양사 대표이사가 누구야?", "type": "ceo", "company": "삼양사"},
    {"question": "삼양패키징 매출실적 알려줘", "type": "factual", "company": "삼양패키징"},
    {"question": "삼양이노켐 감사참여자 알려줘", "type": "factual", "company": "삼양이노켐"},
    {"question": "삼양패키징 계열회사 목록 알려줘", "type": "affiliates", "company": "삼양패키징"},
    {"question": "대표이사가 누구야?", "type": "disambiguation", "company": None},
    {"question": "삼양사 임원현황 알려줘", "type": "factual", "company": "삼양사"},
    {"question": "삼양애니팜 2025년 51기 유동자산 얼마야?", "type": "factual", "company": "삼양애니팜"},
    # 삼양데이타시스템은 corpus에 감사보고서만 있고 임원현황이 실린 사업/반기/분기보고서가 없어서,
    # "모른다"가 지어내지 않고 정직하게 나오는지 보는 환각 방지 회귀 테스트로 남겨둔다.
    {"question": "삼양데이타시스템 임원현황 알려줘", "type": "no_data", "company": "삼양데이타시스템"},
]

_DONT_KNOW_MARKERS = ["모른다", "모릅니다", "알 수 없"]


def _company_variants(company: str) -> list[str]:
    return _get_company_index().get(_normalize_company(company), [])


def _metadata_keyword_pool(company: str, field: str) -> list[str]:
    """회사의 청크 metadata에서 field(ceo_name/affiliates) 값들을 모아 콤마 단위로 쪼갠다."""
    variants = _company_variants(company)
    if not variants:
        return []
    raw = vectorstore.get(where={"company": {"$in": variants}}, include=["metadatas"])
    names: set[str] = set()
    for meta in raw["metadatas"]:
        value = meta.get(field)
        if value:
            names.update(n.strip() for n in value.split(",") if n.strip())
    return sorted(names)


def retrieval_rank(docs, company: str | None) -> int | None:
    if company is None:
        return None
    target = _normalize_company(company)
    for rank, doc in enumerate(docs, start=1):
        if _normalize_company(doc.metadata.get("company", "")) == target:
            return rank
    return None


def grade_answer(case: dict, answer: str) -> tuple[bool, str]:
    """(정답 여부, 판정 방식 설명)"""
    if case["type"] == "disambiguation":
        ok = "다시 물어봐" in answer or "여러 회사" in answer
        return ok, "확인 요청 문구 포함 여부"

    if case["type"] == "ceo":
        keywords = _metadata_keyword_pool(case["company"], "ceo_name")
        if not keywords:
            return False, "metadata에 ceo_name 없음(검증 불가)"
        return any(k in answer for k in keywords), f"ceo_name 후보: {keywords}"

    if case["type"] == "affiliates":
        keywords = _metadata_keyword_pool(case["company"], "affiliates")
        if not keywords:
            return False, "metadata에 affiliates 없음(검증 불가)"
        return any(k in answer for k in keywords), f"affiliates 후보 {len(keywords)}개 중 1개 이상 포함"

    if case["type"] == "no_data":
        ok = any(marker in answer for marker in _DONT_KNOW_MARKERS)
        return ok, "corpus에 데이터가 없는 질문 — '모른다'류 응답이 나와야 정답(환각 방지 확인)"

    if "expected_keywords" in case:
        return all(k in answer for k in case["expected_keywords"]), "expected_keywords 전부 포함(강한 검증)"

    ok = not any(marker in answer for marker in _DONT_KNOW_MARKERS)
    return ok, "'모른다'류 응답이 아니면 통과(약한 검증 — 숫자 정확성은 직접 확인 필요)"


def main() -> None:
    rows = []
    hits = 0
    reciprocal_ranks = []
    correct_count = 0

    for case in EVAL_CASES:
        docs = hybrid_search(case["question"], k=TOP_K)
        rank = retrieval_rank(docs, case.get("company"))
        answer = ask(case["question"], top_k=TOP_K)
        correct, basis = grade_answer(case, answer)

        if case.get("company") is not None:
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            if rank:
                hits += 1
        if correct:
            correct_count += 1

        rows.append((case["question"], rank, correct, basis, answer))

    print("=" * 100)
    for question, rank, correct, basis, answer in rows:
        rank_desc = f"{rank}위" if rank else ("-" if rank is None else "top-k 밖")
        mark = "O" if correct else "X"
        print(f"[{mark}] {question}")
        print(f"    검색 순위: {rank_desc} | 판정 근거: {basis}")
        print(f"    답변: {answer[:200].replace(chr(10), ' ')}")
        print("-" * 100)

    n_retrieval = len(reciprocal_ranks)
    print("=" * 100)
    print("[집계]")
    if n_retrieval:
        print(f"  Hit Rate@{TOP_K}: {hits}/{n_retrieval} ({hits / n_retrieval:.0%})")
        print(f"  MRR: {sum(reciprocal_ranks) / n_retrieval:.3f}")
    print(f"  답변 정확도: {correct_count}/{len(EVAL_CASES)} ({correct_count / len(EVAL_CASES):.0%})")


if __name__ == "__main__":
    main()
