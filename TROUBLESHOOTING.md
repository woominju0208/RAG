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

## 문제 2: 배포는 성공했지만, 챗봇 사용 시 "Failed to fetch" (진행 중)

**상태**: 원인 확정, 해결책 미정.

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
