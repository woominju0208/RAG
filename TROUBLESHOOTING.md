# Render 배포 트러블슈팅 기록

## 문제 1: Render 배포 자체가 Failed로 뜸 (해결됨)

**증상**: Render 대시보드에 배포가 "Failed"로 표시됨 (커밋 `84370f6` 배포 시점).

**로그로 확인한 원인**:
```
INFO: Waiting for application startup.
==> No open ports detected, continuing to scan...
==> Exited with status 3
```

`api.py`의 FastAPI `lifespan` 훅이 서버 기동 전에 `_ingest_if_empty()`를 동기 실행한다.
벡터스토어가 비어있으면(무료 플랜은 Persistent Disk 미지원이라 재배포마다 파일시스템이
초기화됨) 82개 DART XML 전체를 OpenAI로 재임베딩하게 되는데, 이 작업이 Render의 포트
감지 제한시간보다 오래 걸려 배포가 강제 종료됨.

**조치**:
1. 로컬에 이미 만들어져 있던 Chroma 벡터스토어(21,242개 청크, `chroma.sqlite3` 210MB +
   HNSW 인덱스 `.bin` 파일들)를 레포에 **Git LFS**로 커밋 (`chroma_data/`) — GitHub
   100MB 단일 파일 제한 때문에 일반 git이 아닌 LFS 사용.
2. Render 환경변수에 `CHROMA_DIR=chroma_data` 추가 → 배포 시 이미 채워진 데이터를
   읽게 해서 재수집 자체를 건너뛰도록 함.
3. 부수적으로 `api.py`의 CORS `allow_origin_regex`에 붙어있던 trailing slash 버그도 수정
   (`https://food-company-sales-1.onrender.com/` → `https://food-company-sales-1\.onrender\.com`).

**결과**: 배포 성공. 로그에 `[startup] 기존 벡터스토어 사용 (21242개 청크) - 자동 수집
건너뜀.` 확인, `/health` 200 OK, 서비스 "Live" 상태 확인됨.

## 문제 2: 배포는 성공했지만, 챗봇 사용 시 "Failed to fetch" (원인 확정, 임시 우회 중)

**상태**: 근본 원인(무료 플랜 메모리 초과)은 아직 안 고침. 대신 로컬 PC에서
`uvicorn`을 직접 띄우고 **ngrok 고정 도메인**(`https://footsie-habitual-hurdle.ngrok-free.dev`)으로
터널링해서 그걸 실제 백엔드로 임시 사용 중 (→ 문제 3 참고). Render의 RAG 백엔드
배포(`rag-ayud.onrender.com`) 자체는 여전히 이 메모리 문제를 안고 있는 상태.

**증상**: 프론트엔드(`food-company-sales-1.onrender.com`)에서 챗봇에 질문하면
"Failed to fetch" 에러.

**단계별로 확인한 것**:
1. CORS 프리플라이트(OPTIONS)는 정상 — `access-control-allow-origin` 헤더 정확히
   반환됨. CORS 자체는 문제가 아님.
2. 실제 `POST /chat` 요청은 **502 Bad Gateway** 반환 (curl로 재현 확인). 브라우저는
   502 응답에 CORS 헤더가 없으니 이걸 "Failed to fetch"로 표시하는 것 — 근본 원인은
   백엔드가 요청 처리 중 죽는 것.
3. Render 런타임 로그 확인 결과: `OPTIONS /chat 200 OK` 직후 실제 `POST /chat` 로그
   없이 서버가 통째로 재시작됨.
4. Render Events 탭에서 `4:55 PM Instance failed: xtf51` 확정 — 같은 시각. 15분 이상
   무입력이라야 발동하는 "무료 인스턴스 spin down"과는 무관(21초 간격이라 스핀다운
   조건이 안 맞음). 즉 요청 처리 중 인스턴스 자체가 크래시.

**유력한 원인 (코드상 근거)**: `hybrid_search.py`의 `_get_bm25_retriever()` — 회사명
필터가 안 걸리는 질문(예: "test")이면 매 최초 요청마다 `vectorstore.get()`으로
21,242개 청크 전체를 메모리에 끌어와 BM25 인덱스를 새로 빌드함. 이미 Chroma HNSW
인덱스(133MB)도 메모리에 로드된 상태에서, 무료 플랜 RAM 512MB 한도를 넘겨 OOM으로
죽었을 가능성이 높음. (Metrics의 정확한 Memory 그래프 수치는 아직 미확인 — Network
Metrics만 확인됨.)

**아직 결정 안 한 것**: 이 문제를 코드 최적화(BM25를 요청마다 재빌드하지 않도록
캐싱/사전 빌드 등)로 무료 플랜 안에서 해결할지, 아니면 유료 플랜(더 큰 RAM)으로
업그레이드할지 — 다음에 논의해서 정하면 됨.

## 문제 3: ngrok으로 우회 배포 후, 나만 챗봇이 되고 다른 사람은 "Failed to fetch" (해결됨)

**배경**: 문제 2(Render 무료 플랜 메모리 초과)를 임시로 피하려고, 로컬 PC에서
`uvicorn api:app --port 8001`을 띄우고 `ngrok http 8001`로 터널링해서 그 주소를
실제 배포 사이트(`food-company-sales-1.onrender.com`, 프론트엔드)의 백엔드로 쓰기로 함.

**증상**: 본인 컴퓨터/브라우저에서는 챗봇이 정상 응답하는데, 다른 사람이 배포
사이트에서 쓰면 "Failed to fetch".

**원인을 좁혀가는 과정 (앞의 두 가설은 틀렸음)**:
1. *가설 1 — ngrok 무료 플랜의 브라우저 경고 인터스티셜*: ngrok 터널은 방문자가
   직접 경고 페이지를 클릭해서 넘겨야 우회 쿠키가 생기고, 그 쿠키가 없는 사람은
   API 호출도 막힌다고 추정. → curl로 `ngrok-skip-browser-warning` 헤더 유무를
   비교 테스트했더니 **헤더 없이도 정상 응답** — 이 가설은 틀림 (기각).
2. *가설 2 — ngrok URL이 재시작마다 바뀌어 프론트 설정이 예전 URL을 가리킴*:
   사용자가 이미 ngrok 고정 도메인(`ngrok-free.dev`)으로 바꿔서 이 문제 자체는
   해결된 상태였음. 그런데도 실패가 계속됨 → 이것도 근본 원인은 아니었음.
3. *실제 원인 발견*: 실패하는 사이트에서 브라우저 개발자도구 Network/Console
   탭을 직접 확인하니, 요청이 ngrok 주소가 아니라 **`http://localhost:8001/chat`**로
   가고 있었음 (`Access to fetch ... has been blocked ... loopback address space`,
   `net::ERR_CONNECTION_REFUSED`). `localhost`는 요청을 보낸 사람 자신의 컴퓨터를
   가리키므로, 본인 PC에서만 우연히 그 주소에 진짜 서버가 떠 있어 성공했던 것.
4. 프론트엔드 코드(`dashboard/frontend/src/api/ragClient.ts`)를 확인해보니
   `const RAG_BASE_URL = import.meta.env.VITE_RAG_API_BASE_URL ?? "http://localhost:8001"` —
   코드 자체는 정상. 즉 배포된 JS 번들이 `VITE_RAG_API_BASE_URL`이 설정되기 *이전*
   시점에 빌드된 예전 산출물이라는 뜻이었음 (Vite 환경변수는 런타임이 아니라
   빌드 시점에 코드에 박제됨).
5. **결정적 원인**: `VITE_RAG_API_BASE_URL`을 Render에서 설정했는데, 그 위치가
   프론트엔드 서비스(`food-company-sales-1`)가 아니라 **RAG 백엔드 서비스**의
   Environment 탭이었음 (`CHROMA_DIR`/`OPENAI_API_KEY`와 나란히 들어가 있었음).
   Vite 환경변수는 프론트엔드 자신을 빌드할 때만 의미가 있어서, 엉뚱한(백엔드)
   서비스에 넣은 값은 프론트 빌드에 전혀 반영되지 않았던 것.

**조치**:
1. `VITE_RAG_API_BASE_URL=https://footsie-habitual-hurdle.ngrok-free.dev`를
   **프론트엔드 서비스(`food-company-sales-1`)**의 Environment 탭으로 옮겨서 추가.
2. 그 프론트엔드 서비스에서 Manual Deploy → **Clear build cache & deploy**로
   완전히 새로 빌드.

**결과**: 다른 컴퓨터에서도 챗봇 정상 동작 확인됨.

**참고**: 이 우회 구성은 로컬 PC의 `uvicorn`+`ngrok`이 계속 켜져 있어야만 동작한다.
컴퓨터를 끄거나 두 프로세스를 종료하면 챗봇 전체가 다시 죽는다. 근본 해결은 문제 2
(Render 무료 플랜 메모리 초과)를 고쳐서 Render 자체 배포(`rag-ayud.onrender.com`)로
`VITE_RAG_API_BASE_URL`을 되돌리는 것.
