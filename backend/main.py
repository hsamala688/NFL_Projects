import logging

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from agent.respond import ask_async, AnalystResponse
from backend import cache
from backend.config import AGENT_API_KEY, FRONTEND_ORIGINS
from backend.schemas import AskRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="NFL Analyst API", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    if not AGENT_API_KEY or not api_key or api_key != AGENT_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


_AUTH = [Security(_require_api_key)]


@app.post("/ask", response_model=AnalystResponse, dependencies=_AUTH)
async def ask(request: AskRequest) -> AnalystResponse:
    try:
        return await ask_async(request.question)
    except Exception as exc:
        msg = str(exc)
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code == 429 or "429" in msg or "quota" in msg.lower() or "resource_exhausted" in msg.lower():
            raise HTTPException(status_code=429, detail="Rate limit reached.")
        if code == 503 or "503" in msg or "unavailable" in msg.lower():
            raise HTTPException(status_code=503, detail="AI service temporarily unavailable.")
        logger.exception("Unhandled error in /ask")
        raise HTTPException(status_code=500, detail="An error occurred processing your question.")


@app.get("/demo/questions", dependencies=_AUTH)
async def demo_questions() -> list[str]:
    return cache.questions()


@app.post("/demo/ask", response_model=AnalystResponse, dependencies=_AUTH)
async def demo_ask(request: AskRequest) -> AnalystResponse:
    cached = cache.lookup(request.question)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Question not in demo cache. Use the live /ask endpoint.",
        )
    return AnalystResponse(**cached)
