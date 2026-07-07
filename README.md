# RAG (문서 기반 질의응답)

LangChain + OpenAI 임베딩 + Chroma(로컬 벡터 DB)로 만든 RAG 파이프라인입니다.
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

1. 문서를 `data/`에 넣습니다.
   - 일반 텍스트: `data/` 바로 아래에 `.txt` 파일로.
   - DART 공시 XML(사업보고서/반기보고서 등 `dart3.xsd` 스키마): `data/` 하위 아무 폴더(예: `data/dart_xml/`)에 `.xml`로 넣으면 재귀적으로 찾습니다. `dart_parser.py`가 SECTION 태그 구조를 따라가며 섹션 제목을 breadcrumb 형태(`I. 회사의 개요 > 1. 회사의 개요`)로 살려서 청크 metadata에 저장합니다.
2. 문서를 임베딩해서 저장:
   ```bash
   python ingest.py
   ```
   같은 파일을 다시 넣고 실행하면 기존 청크를 덮어씁니다 (upsert).
3. 질문하기:
   ```bash
   python query.py "이 보고서에서 매출에 영향을 준 지표는?"
   ```

## 데이터 확인 (노트북)

`explore_data.ipynb`에서 저장된 청크, 유사도 검색 결과, DART XML 파싱 결과를 직접 확인할 수 있습니다.

```bash
pip install -r requirements-dev.txt
```

VSCode에서 노트북을 열고 커널로 `.venv`를 선택한 뒤 셀을 위에서부터 실행하세요.

## 구조

```
rag_core.py      LangChain 임베딩/LLM/텍스트 스플리터/Chroma 벡터스토어 공통 객체
dart_parser.py   DART 공시 XML 전용 파서: SECTION 트리를 breadcrumb 있는 Document 목록으로 변환
hybrid_search.py 벡터 검색 + BM25(키워드) 검색을 합치는 하이브리드 검색
ingest.py        data/*.txt, data/**/*.xml -> Document 변환 -> 청크 분할 -> Chroma 저장
query.py         질문 -> hybrid_search로 검색 -> LCEL 체인(prompt | llm)으로 답변 생성
data/            원본 문서 (git에는 커밋 안 됨)
```

`rag_core.py`가 만드는 공통 객체:
- `embeddings` — `OpenAIEmbeddings` (text-embedding-3-small)
- `llm` — `ChatOpenAI` (gpt-4o-mini)
- `text_splitter` — `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100)
- `vectorstore` — `Chroma` (embeddings와 연결된 로컬 벡터스토어)

## 검색 방식 (하이브리드: 의미 검색 + 키워드 검색)

`vectorstore.similarity_search()`만 쓰면 의미가 비슷한 문서를 찾아주긴 하지만, "회사명" 같은 고유명사가 질문에 있어도 그 회사 문서를 정확히 우선하지는 않습니다 (예: 여러 계열사 문서가 섞여 있으면 "대표이사"라는 단어가 자주 나오는 다른 회사 문서가 먼저 걸릴 수 있음). `hybrid_search.py`는 이를 보완합니다.

1. **회사명 자동 감지** — 저장된 청크의 `company` 메타데이터를 정규화(`(주)`/`주식회사`/공백 제거)해서 목록을 만들고, 질문에 그 회사명이 들어있으면 해당 회사의 청크로만 검색 범위를 좁힙니다.
2. **BM25(키워드) 검색** — 한글 조사(삼양패키징 vs 삼양패키징의) 때문에 단어 단위 토큰화가 잘 안 맞는 문제를 피하려고, 음절 2-gram으로 토큰화한 BM25를 씁니다.
3. **벡터(의미) 검색**과 **BM25 검색** 결과를 RRF(Reciprocal Rank Fusion)로 합쳐서 최종 top-k를 뽑습니다.

회사명이 질문에 없으면 1번 단계 없이 전체 문서에서 하이브리드 검색만 수행합니다.

**벡터 인덱스 저장 위치**: 기본값은 `~/.rag_project_chroma` (프로젝트 폴더 밖, 사용자 홈 디렉터리)입니다. `CHROMA_DIR` 환경변수로 바꿀 수 있습니다. 프로젝트 폴더 안(`.chroma/`)에 두지 않는 이유는, chromadb의 Rust 기반 HNSW 인덱스가 경로에 비ASCII 문자(한글 폴더명 등)가 섞이면 인덱스 파일을 제대로 못 만드는 버그가 있어서입니다 — 데이터는 sqlite에 정상 저장되지만 이후 조회 시 `Error loading hnsw index` 에러가 납니다. 프로젝트 경로 자체에 한글이 들어있는 한(`바탕 화면` 등) 이 문제를 피하려면 인덱스 디렉터리를 영문 경로로 분리해야 합니다.

## 다음 단계 후보

- PDF → 텍스트 자동 변환 전처리 추가
- 청크 크기/오버랩 튜닝
- 여러 문서에 걸친 답변 시 출처(source) 표시 강화
- FastAPI로 감싸서 웹에서 질의하는 엔드포인트 추가
