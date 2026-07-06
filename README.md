# RAG (문서 기반 질의응답)

OpenAI 임베딩 + Chroma(로컬 벡터 DB)로 만든 최소 동작 RAG 파이프라인입니다.
`data/`에 넣은 텍스트 문서를 검색 가능하게 저장하고, 질문하면 관련 부분을 찾아 GPT로 답을 생성합니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env
# .env에 OPENAI_API_KEY를 채워 넣습니다
```

## 사용법

1. `data/` 폴더에 `.txt` 파일로 문서를 넣습니다 (PDF/XML 등은 미리 텍스트로 변환 필요).
2. 문서를 임베딩해서 저장:
   ```bash
   python ingest.py
   ```
   같은 파일을 다시 넣고 실행하면 기존 청크를 덮어씁니다 (upsert).
3. 질문하기:
   ```bash
   python query.py "이 보고서에서 매출에 영향을 준 지표는?"
   ```

## 구조

```
rag_core.py   임베딩/청크 분할/Chroma 연결 공통 함수
ingest.py     data/*.txt -> 청크 분할 -> 임베딩 -> Chroma 저장
query.py      질문 -> 임베딩 -> 유사도 검색(top-k) -> GPT 답변 생성
data/         원본 문서 (git에는 커밋 안 됨)
.chroma/      Chroma가 만드는 로컬 벡터 저장 파일 (git에는 커밋 안 됨)
```

## 다음 단계 후보

- PDF/XML(DART 공시 등) → 텍스트 자동 변환 전처리 추가
- 청크 크기/오버랩 튜닝, 문단/섹션 단위 분할로 개선
- 여러 문서에 걸친 답변 시 출처(source) 표시 강화
- FastAPI로 감싸서 웹에서 질의하는 엔드포인트 추가
