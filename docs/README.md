# Azetta — Developer Documentation

Comprehensive developer guide for the `azetta-backend` (RAG anti-lissage for Amazigh craft knowledge).

This document covers architecture, setup, ingestion, runtime, API reference, internals (MMR + constitution), data model, contribution moderation flow, deployment notes, metrics, and troubleshooting. Diagrams are provided as Mermaid blocks and can be rendered by GitHub/VS Code Preview.

**Quick links**
- Repository root: `README.md` (short summary)
- Main API: `app/main.py`
- Ingest: `app/ingest.py`
- RAG engine: `app/rag.py`
- Config: `app/config.py`
- Metrics: `app/metrics.py`

---

## 1. Overview

Azetta is a retrieval-augmented system designed to answer questions about Amazigh crafts and traditions while avoiding cultural "lissage" (smoothing). It uses:

- LlamaIndex (core) for node and index abstractions
- Gemini (Google GenAI) as LLM and embeddings
- Qdrant as the vector store (embedded by default)
- FastAPI for the HTTP API

The system's core idea: retrieve multiple candidate sources, apply Maximal Marginal Relevance (MMR) to maximize diversity, and synthesize an answer under a strict "cultural constitution" system prompt so responses remain faithful and sourced.

---

## 2. Quickstart (developer)

Prerequisites: Python >= 3.12, project dependencies installed via `uv` (see `pyproject.toml`). Set your Gemini API key in `.env`.

Commands (copy-paste):

```bash
uv sync
cp .env.example .env
# Edit .env and set GEMINI_API_KEY or GOOGLE_API_KEY
uv run python -m app.ingest          # build or load index
uv run uvicorn app.main:app --port 8000
```

Open the interactive API docs at http://localhost:8000/docs

---

## 3. Installation & Env

- Install pinned deps: `uv sync`.
- Copy `.env.example` → `.env` and fill `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

Important config lives in `app/config.py`. Key environment-aware knobs:

- `GEMINI_API_KEY`, `AZETTA_LLM_MODEL`, `AZETTA_EMBED_MODEL`
- `EMBED_DIM` (must match the Qdrant collection vector size)
- `COLLECTION_NAME` (Qdrant collection)
- Retrieval and anti-lissage: `OVERFETCH_K`, `SIMILARITY_TOP_K`, `MMR_LAMBDA`, `ROUTER_SCORE_THRESHOLD`

---

## 4. Ingesting the Corpus

Corpus files: one JSON fiche per file under the `corpus/` directory. Each fiche must include the schema:

- `id`, `categorie`, `titre`, `contenu`, `region`, `termes_amazighs` (array), `elements_culturels` (array), `source`, `fiabilite`

To (re)build the index:

```bash
uv run python -m app.ingest         # uses embedded Qdrant; builds if missing
uv run python -m app.ingest --force # drop & rebuild collection
```

Implementation notes:

- `app/ingest.py::build_index()` validates every JSON fiche and converts each fiche to a single `TextNode` (no chunking). `id_` is a deterministic UUID (`fiche_uuid`) derived from the fiche id so re-ingestion upserts instead of duplicating.
- Embedded Qdrant uses `data/qdrant_data/` by default. To use a remote Qdrant server, replace `get_client()` in `app/ingest.py` with `QdrantClient(url="http://localhost:6333")`.

---

## 5. Run the API

Run a single worker (embedded Qdrant takes an exclusive lock — do not use `--reload`):

```bash
uv run uvicorn app.main:app --port 8000
```

Endpoints (primary):

- `POST /ask` — question -> `{response, source_nodes, metrics}`
- `POST /compare` — runs RAG vs baseline LLM (for lissage detection)
- `POST /chat` — routed chat with per-chat memory
- `POST /feedback` — record user votes/corrections (HITL loop)
- `POST /contributions` — submit a new fiche candidate
- `GET /contributions` — moderation queue listing
- `POST /contributions/{id}/moderate` — approve/reject a contribution (approve triggers re-index via `app/ingest.promote_contribution`)

Refer to `app/main.py` for Pydantic schemas and payload validation.

### Example: `POST /ask`

Request example:

```json
{ "question": "Que signifie le losange dans le tissage amazigh ?" }
```

Response shape (example):

```json
{
  "response": "...",
  "source_nodes": [ { "id": "...", "titre": "...", "region": "...", "score": 0.83 } ],
  "metrics": { "cultural_coverage": { "percent": 42.0 }, "other_metrics": {} }
}
```

---

## 6. RAG internals

The RAG pipeline lives in `app/rag.py`. Key components:

- Retrieval: uses `index.as_retriever(similarity_top_k=OVERFETCH_K)` to overfetch candidates.
- Diversity re-ranking: `mmr_rerank(query_emb, cand_embs, k, lambda_)` implements Maximal Marginal Relevance.
- Synthesis: builds the system prompt called the `CULTURAL_CONSTITUTION` (an imperative set of rules) and synthesizes the final answer over the selected nodes.

Routing (agentic): `ask_routed()` first gates using the bare question to decide whether to answer directly with the LLM (no retrieval) or run the cultural RAG path, based on `ROUTER_SCORE_THRESHOLD`.

Important functions and references:

- `app/rag.py::mmr_rerank` — MMR implementation (numpy).
- `app/rag.py::CULTURAL_CONSTITUTION` — the system prompt enforcing cultural fidelity.
- `app/rag.py::ask()` — RAG answer flow.
- `app/rag.py::ask_baseline()` — baseline LLM answer (no retrieval/constitution) used by `/compare`.

---

## 7. Data model

- Source artifacts are JSON fiches under `corpus/`.
- Each fiche → a single `TextNode` with metadata (fiche_id, titre, region, source, categorie, termes_amazighs, elements_culturels, fiabilite).
- Embeddings are created by the configured Gemini embedding model; dimensionality is pinned by `EMBED_DIM` in `app/config.py` and must match the Qdrant collection.

---

## 8. Contribution moderation flow

Flow summary:

1. Public user posts `POST /contributions` with a candidate fiche.
2. `app/moderation.py::screen()` runs auto-filters (spam, duplicate detection using embeddings, heuristic checks).
3. `app/moderation.py::decide_status()` returns `pending` or `auto_rejected`.
4. Experts call `POST /contributions/{id}/moderate` to `approve` or `reject`.
5. `approve` triggers `app/ingest.promote_contribution()` which writes a new `corpus/<id>.json` and upserts the node into the live Qdrant collection.

---

## 9. Deployment & Embedded Qdrant notes

- Embedded (default): Qdrant persists to `data/qdrant_data/` and is accessed with `QdrantClient(path=config.QDRANT_PATH)`. The embedded store holds an exclusive file lock — run a single worker and avoid `uvicorn --reload`.
- Server mode: use a networked Qdrant and update `get_client()` in `app/ingest.py` to `QdrantClient(url='http://<host>:6333')`. In server mode you can run multiple app workers.

---

## 10. Metrics & verification

`app/metrics.py` exposes helpers used in API responses to compute cultural coverage and related metrics. Use `/compare` to measure delta between the RAG path and baseline LLM.

Recommended verification steps after ingest:

1. Run a few on-topic `POST /ask` requests and verify `source_nodes` are non-empty.
2. Check `metrics.cultural_coverage.percent` is reasonable for known topics.
3. For duplicate-detection tuning, adjust `DUPLICATE_SCORE_THRESHOLD` in `app/config.py`.

---

## 11. Troubleshooting

- If LlamIndex attempts to contact OpenAI: ensure `app.config.configure_settings()` runs early (it is called at app startup and in `ingest.build_index`) and that `GEMINI_API_KEY` is set.
- If `app.ingest` fails with Qdrant path issues: ensure `data/` is writable and no other process holds the embedded DB lock.
- If embeddings dimensions mismatch: confirm `EMBED_DIM` in `app/config.py` matches the embedding model output and the Qdrant collection vector size. Recreate the collection with `--force` if you change embed dims.

---

## 12. Diagrams (Mermaid)

### 12.1 Architecture (components & data flow)

```mermaid
graph LR
  A[Client] -->|HTTP| B[FastAPI `app/main.py`]
  B --> C[RAG service `app/rag.py`]
  C --> D[Retriever (LlamaIndex index)]
  D --> E[Qdrant (embedded / remote)]
  C --> F[Gemini LLM & Embeddings]
  B --> G[Feedback DB `app/feedback.py`]
  B --> H[Contributions DB `app/contributions.py`]
```

### 12.2 Sequence: POST /ask → RAG → Response

```mermaid
sequenceDiagram
  participant Client
  participant API
  participant RAG
  participant Qdrant
  participant LLM

  Client->>API: POST /ask {question}
  API->>RAG: ask(question)
  RAG->>Qdrant: retriever.retrieve(overfetch_k)
  Qdrant-->>RAG: candidate nodes
  RAG->>LLM: embed query + candidate texts
  LLM-->>RAG: embeddings
  RAG->>RAG: mmr_rerank -> select top-k
  RAG->>LLM: synthesize with CULTURAL_CONSTITUTION
  LLM-->>RAG: answer
  RAG-->>API: {response, sources, metrics}
  API-->>Client: HTTP 200
```

### 12.3 Data model flow (fiche → TextNode → vector)

```mermaid
flowchart LR
  Fiche[corpus/*.json] -->|validated| Builder[app/ingest.build_nodes]
  Builder --> TextNode[TextNode(id_, text, metadata)]
  TextNode -->|embed| Vector[embedding vector]
  Vector --> Qdrant[Qdrant collection 'azetta_fiches']
```

### 12.4 Deployment (embedded vs server Qdrant)

```mermaid
flowchart TB
  Dev[Developer laptop] -->|embedded| App[FastAPI + embedded Qdrant]
  App --> Data[data/qdrant_data]
  Prod[Production] -->|remote qdrant| App2[FastAPI]
  App2 -->|http| QdrantServer[Qdrant (remote)]
```

### 12.5 Contribution moderation flow

```mermaid
flowchart LR
  User -->|POST /contributions| API[app/main.py]
  API --> Moderation[app/moderation.screen]
  Moderation -->|flags| Decide[app/moderation.decide_status]
  Decide -->|pending/auto_rejected| DB[app/contributions.insert]
  Expert -->|list| API
  Expert -->|moderate approve| API
  API -->|promote| Ingest[app/ingest.promote_contribution]
  Ingest -->|write| Corpus[corpus/<id>.json]
  Ingest -->|upsert| Qdrant
```

---

## 13. Contributing

If you want to extend the backend:

1. Fork the repo and create a feature branch.
2. Run `uv sync` and add tests where appropriate.
3. Keep changes small and focused: update `docs/README.md` when adding features.

---

## 14. File references

- API: `app/main.py`
- Ingest / index: `app/ingest.py`
- RAG core: `app/rag.py`
- Config: `app/config.py`
- Metrics: `app/metrics.py`
- Moderation: `app/moderation.py`

---

End of developer documentation.
