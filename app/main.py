"""FastAPI app for CULTINI — MVP endpoints.

  POST /ask                          question -> response + sources + metrics
  POST /chat                         routed, context-aware chat (per-chat memory), SSE stream
  POST /feedback                     record a vote and/or correction (HITL loop)
  POST /contributions                submit a contribution -> auto-filter + queue
  GET  /contributions                moderation queue (expert)
  POST /contributions/{id}/moderate  approve (promote to fiche + re-index) | reject
  GET  /stats                        HITL dashboard numbers + corpus stats

Run (single worker, NO --reload — embedded Qdrant takes an exclusive lock):
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import config, contributions, conversations, feedback, ingest, metrics, moderation, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm everything at startup so the first request isn't slow and
    # configuration errors surface immediately.
    config.configure_settings()
    feedback.init_db()
    conversations.init_db()
    contributions.init_db()
    rag.get_referential()
    rag.get_index()
    yield


app = FastAPI(title="CULTINI", description="RAG anti-lissage — artisanat amazigh", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,  # any localhost port (Flutter web dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceNode(BaseModel):
    id: str | None = None
    titre: str | None = None
    region: str | None = None
    source: str | None = None
    categorie: str | None = None
    fiabilite: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    response: str
    source_nodes: list[SourceNode]
    metrics: dict


class CompareResponse(BaseModel):
    question: str
    cultini: AskResponse
    baseline: AskResponse
    coverage_delta: float  # cultini - baseline, en points de cultural_coverage.percent


class ChatRequest(BaseModel):
    chat_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1)
    ai_response: str = Field(..., min_length=1)
    vote: str | None = Field(None, pattern="^(up|down)$")
    reason: str | None = None
    correction: str | None = None


class ContributionRequest(BaseModel):
    titre: str = Field(..., min_length=3)
    categorie: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)        # wilaya / région d'origine
    contenu: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    contributor_name: str | None = None


class ContributionResponse(BaseModel):
    id: int
    status: str               # 'pending' | 'auto_rejected'
    accepted: bool            # True when it entered the human queue (status == pending)
    message: str
    flags: dict


class ModerationRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|reject)$")


# --- Endpoints ---
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = rag.ask(req.question)
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**result)


@app.post("/compare", response_model=CompareResponse)
def compare(req: AskRequest) -> CompareResponse:
    """Détecteur de lissage : même question -> CULTINI (RAG) vs Gemini brut, côte à côte."""
    try:
        azetta = rag.ask(req.question)
        baseline = rag.ask_baseline(req.question)
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    delta = round(
        azetta["metrics"]["cultural_coverage"]["percent"]
        - baseline["metrics"]["cultural_coverage"]["percent"],
        1,
    )
    return CompareResponse(
        question=req.question,
        cultini=AskResponse(**azetta),
        baseline=AskResponse(**baseline),
        coverage_delta=delta,
    )


def _sse(event: dict) -> str:
    """Serialise one SSE ``data:`` frame (single-line JSON + blank-line terminator)."""
    return "data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n\n"


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Routed, context-aware chat streamed as SSE: meta -> token* -> done.

    Agentic gate (RAG vs bare LLM) + per-chat memory. The ``meta`` event carries
    the route, router score and (cultural path) the sources; ``token`` events
    carry incremental text deltas; ``done`` carries the metrics computed on the
    full answer. Turns are persisted AFTER the stream so stored text matches what
    the client received.
    """
    history = conversations.get_history(req.chat_id, limit=config.CHAT_HISTORY_TURNS)

    def event_stream():
        try:
            meta, token_iter = rag.ask_routed_stream(req.question, history)
        except Exception as exc:  # surface a clean error to the client
            yield _sse({"type": "error", "detail": str(exc)})
            return

        yield _sse({"type": "meta", "chat_id": req.chat_id, **meta})

        chunks: list[str] = []
        try:
            for delta in token_iter:
                if not delta:
                    continue
                chunks.append(delta)
                yield _sse({"type": "token", "delta": delta})
        except Exception as exc:
            yield _sse({"type": "error", "detail": str(exc)})
            return

        full_text = "".join(chunks)
        conversations.append_turn(req.chat_id, "user", req.question)
        conversations.append_turn(req.chat_id, "assistant", full_text, route=meta["route"])
        yield _sse(
            {"type": "done", "metrics": metrics.response_metrics(full_text, list(rag.get_referential()))}
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx / Render)
            "Connection": "keep-alive",
        },
    )


@app.get("/chat/{chat_id}/history")
def chat_history(chat_id: str) -> dict:
    return {"chat_id": chat_id, "turns": conversations.get_history(chat_id, limit=50)}


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    if req.vote is None and not (req.correction or req.reason):
        raise HTTPException(status_code=400, detail="Provide a vote and/or a correction/reason.")
    fid = feedback.insert_feedback(
        question=req.question,
        ai_response=req.ai_response,
        vote=req.vote,
        reason=req.reason,
        correction=req.correction,
    )
    return {"status": "ok", "id": fid}


# --- Contribution flow ---
@app.post("/contributions", response_model=ContributionResponse)
def submit_contribution(req: ContributionRequest) -> ContributionResponse:
    """Public submission endpoint (the 'Contribuer' button).

    Runs the auto-filter (spam / doublon / contenu IA), enqueues the result in
    the moderation queue, and tells the client whether it was accepted into the
    queue or auto-rejected.
    """
    flags = moderation.screen(req.titre, req.contenu)
    status, message = moderation.decide_status(flags)
    contrib_id = contributions.insert_contribution(
        titre=req.titre,
        categorie=req.categorie,
        region=req.region,
        contenu=req.contenu,
        source=req.source,
        contributor_name=req.contributor_name,
        status=status,
        flags=flags,
    )
    return ContributionResponse(
        id=contrib_id,
        status=status,
        accepted=status == "pending",
        message=message,
        flags=flags,
    )


@app.get("/contributions")
def list_contributions(status: str | None = None, limit: int = 100) -> dict:
    """Moderation queue listing for an expert (optionally filter by status)."""
    return {
        "contributions": contributions.list_contributions(status=status, limit=limit),
        "stats": contributions.get_stats(),
    }


@app.post("/contributions/{contrib_id}/moderate")
def moderate_contribution(contrib_id: int, req: ModerationRequest) -> dict:
    """Expert validation. ``approve`` promotes the submission to a documented
    fiche and re-indexes Qdrant; ``reject`` just records the decision."""
    contrib = contributions.get_contribution(contrib_id)
    if contrib is None:
        raise HTTPException(status_code=404, detail="Contribution introuvable.")
    if contrib["status"] not in ("pending", "auto_rejected"):
        raise HTTPException(
            status_code=409,
            detail=f"Déjà modérée (statut: {contrib['status']}).",
        )

    if req.decision == "reject":
        contributions.set_status(contrib_id, "rejected")
        return {"status": "rejected", "id": contrib_id}

    try:
        fiche = ingest.promote_contribution(contrib, rag.get_index())
    except Exception as exc:  # surface indexing/embedding failures cleanly
        raise HTTPException(status_code=500, detail=f"Échec de l'indexation : {exc}") from exc
    # New fiche changed the corpus → drop cached referential so coverage reflects it.
    rag.get_referential.cache_clear()
    contributions.set_status(contrib_id, "approved", fiche_id=fiche["id"])
    return {"status": "approved", "id": contrib_id, "fiche_id": fiche["id"]}


@app.get("/stats")
def stats() -> dict:
    fiches = ingest.load_fiches()
    a_verifier = sum(1 for f in fiches if f.get("fiabilite") == "a_verifier")
    return {
        "corpus": {
            "total_fiches": len(fiches),
            "a_verifier": a_verifier,
            "referential_size": len(rag.get_referential()),
        },
        "feedback": feedback.get_stats(),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
