# cultini_AI —  RAG backend

RAG service for **Cultini**, focused on documenting and amplifying Amazigh
(Berber) craftsmanship without flattening it. It answers questions strictly from a
curated corpus under a "cultural constitution" prompt, measures how much cultural
detail each answer preserves, and runs a moderated pipeline for community
contributions. Package name: `azetta-backend`.

> **Full developer documentation:** [docs/README.md](docs/README.md) — architecture,
> RAG internals, the moderation flow, tuning, deployment, and the complete config
> reference. This file is a short overview and quickstart.

## What it does

- **Constitution-guided RAG** — answers only from the corpus; keeps Amazigh terms,
  always cites region and source, and refuses to "lisser" (smooth away) regional
  differences.
- **Diversity re-ranking** — a hand-rolled MMR step over retrieved candidates so
  answers stay varied rather than collapsing to a generic mean.
- **Agentic router** — if retrieval confidence is below a threshold, it answers with
  the bare LLM (clearly flagged) instead of forcing a weak corpus answer.
- **Moderated contributions** — public submissions pass an auto-filter
  (spam / duplicate / AI-generated checks, all fail-open), queue for expert review,
  and on approval are promoted into the corpus and re-indexed.

## Tech stack

- **FastAPI** + **uvicorn**
- **LlamaIndex** for the RAG plumbing
- **Qdrant** vector store (embedded on-disk by default; networked in production)
- **Google Gemini** for embeddings and generation (via `llama-index-*-google-genai`)
- **SQLite** for the contribution / feedback / conversation stores
- **uv** for dependency management (`uv.lock`), Python `>=3.12`

## Quickstart

```bash
uv sync
cp .env.example .env
# set GEMINI_API_KEY in .env
uv run python -m app.ingest                       # build the Qdrant index from corpus/
uv run uvicorn app.main:app --port 8000           # run the API (single worker)
```

Open http://localhost:8000/docs for the interactive Swagger explorer.

Notes:

- Run with a **single uvicorn worker** — the default embedded Qdrant takes an
  exclusive on-disk lock. For multiple workers, point at a networked Qdrant
  (`QDRANT_URL` / `QDRANT_API_KEY`). Do not use `--reload` for the same reason.
- Re-ingest after changing the corpus: `uv run python -m app.ingest --force`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/ask` | RAG answer with source nodes and cultural-coverage metrics. |
| `POST` | `/compare` | Side-by-side Cultini RAG vs bare LLM, with a coverage delta (lissage detector). |
| `POST` | `/chat` | Routed, context-aware chat with per-`chat_id` memory; streamed as Server-Sent Events. |
| `GET` | `/chat/{chat_id}/history` | Retrieve a conversation's turns. |
| `POST` | `/feedback` | Human feedback (vote up/down, reason, correction). |
| `POST` | `/contributions` | Submit a contribution; runs the auto-filter and queues it for moderation. |
| `GET` | `/contributions` | List the moderation queue (expert view). |
| `POST` | `/contributions/{id}/moderate` | Approve (promote to corpus) or reject a contribution. |
| `GET` | `/stats` | Corpus and feedback statistics. |
| `GET` | `/health` | Health check. |

## Configuration

Only `GEMINI_API_KEY` is required; everything else has a sensible default. A few you
may want to tune:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Required. Google Gemini API key. |
| `AZETTA_LLM_MODEL` | `gemini-2.5-flash` | Generation model. |
| `AZETTA_DUPLICATE_THRESHOLD` | `0.85` | Cosine similarity above which a contribution is flagged as a duplicate. |
| `AZETTA_ROUTER_THRESHOLD` | `0.6` | Retrieval score below which the router answers with the bare LLM. |
| `QDRANT_URL` / `QDRANT_API_KEY` | — | Set to use a networked Qdrant instead of the embedded store. |

See [docs/README.md](docs/README.md) for the complete environment-variable table,
the fiche/corpus schema, the MMR and moderation internals, and deployment guidance.
