"""RAG 챗봇 API 서버 (FastAPI).

프로젝트 사이트가 이 서버에 질문을 HTTP로 보내면 hybrid_search + LLM으로 답을 만들어 돌려준다.
사이트가 쓰는 로컬 DB는 이 RAG의 DART 데이터와 완전히 별개이므로 여기서 전혀 건드리지 않는다 —
DART 데이터는 이 프로젝트의 Chroma 벡터스토어에만 있고, 사이트는 이 엔드포인트만 호출하면 된다.

실행:
    uvicorn api:app --host 0.0.0.0 --port 8001
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from query import ask
from rag_core import vectorstore


def _ingest_if_empty() -> None:
    """배포 환경(영구 볼륨)이 비어 있는 최초 1회에만 자동으로 data/ 문서를 수집한다.
    이미 청크가 있으면(=이전 배포에서 이미 수집됨) 건너뛴다 — 안 그러면 재배포할 때마다
    전체를 다시 임베딩하게 되어 시간/비용이 매번 든다. count()는 전체 문서를 안 불러오고
    개수만 세므로 매 시작마다 호출해도 가볍다."""
    count = vectorstore._collection.count()
    if count > 0:
        print(f"[startup] 기존 벡터스토어 사용 ({count}개 청크) - 자동 수집 건너뜀.")
        return

    print("[startup] 벡터스토어가 비어 있어 최초 1회 자동 수집을 시작합니다. "
          "문서 양에 따라 수 분~수십 분 걸릴 수 있습니다...")
    from ingest import main as ingest_main
    ingest_main()
    print("[startup] 자동 수집 완료.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 이 블록이 끝나기 전까지(=최초 배포에서 자동 수집이 끝나기 전까지) 서버가 요청을
    # 받지 않는다. 플랫폼의 헬스체크 타임아웃이 짧으면 첫 배포에서 실패로 오인될 수 있으니
    # 배포 플랫폼의 헬스체크 유예 시간을 넉넉히 잡아야 한다.
    _ingest_if_empty()
    yield


app = FastAPI(title="DART RAG 챗봇 API", lifespan=lifespan)

# 프로덕션 도메인은 콤마로 구분해 환경변수에 넣는다 (예: ALLOWED_ORIGINS=https://example.com).
# 로컬 개발 중에는 Vite가 포트를 바꿔가며 뜨므로(5173, 5174, ...) localhost/127.0.0.1은
# 포트 상관없이 정규식으로 항상 허용한다 (대시보드 백엔드의 CORS 설정과 동일한 패턴).
_allowed_origins = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    # allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_origin_regex=r"https://food-company-sales-1.onrender.com/",
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # 일반 def로 선언해야 FastAPI가 스레드풀에서 실행한다. ask()는 OpenAI 호출을 동기(블로킹)로
    # 하므로, async def로 선언하면 응답이 오는 동안 이벤트 루프 전체가 멈춰 다른 요청을 못 받는다.
    return ChatResponse(answer=ask(req.question))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
