# cultini — Backend

RAG anti-lissage sur l'artisanat amazigh. Stack : **LlamaIndex + Gemini + Qdrant + FastAPI**, géré avec **uv**.

## Setup

```bash
uv sync                       # install pinned deps from uv.lock
cp .env.example .env          # then put your GEMINI_API_KEY in .env
```

Get a Gemini API key at https://aistudio.google.com/apikey.

Check which models your key can use (optional):

```bash
uv run python -m app.list_models
```

## Ingest the corpus

Embeds the `corpus/*.json` fiches into an embedded Qdrant store (`data/qdrant_data/`).
Persistent — only re-embeds when the collection is missing or you pass `--force`.

```bash
uv run python -m app.ingest          # build once / load if present
uv run python -m app.ingest --force  # drop and rebuild
```

## Run the API

> Embedded Qdrant takes an exclusive file lock — run a **single worker, no `--reload`**.

```bash
uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs.

| Endpoint | Rôle |
|---|---|
| `POST /ask` | question → réponse + sources + métriques anti-lissage |
| `POST /feedback` | enregistre un vote / une correction (boucle HITL) |
| `GET /stats` | chiffres du dashboard HITL + stats corpus |

## Corpus

One JSON file per fiche in `corpus/`. All seed fiches are flagged
`fiabilite: "a_verifier"` — they are drawn from broadly known references and
**must be validated/corrected by native-speaker experts**. That validation is
the first turn of the HITL loop; once verified, flip the flag to `documentee`
and re-ingest with `--force`.

Fiche schema: `id`, `categorie`, `titre`, `contenu`, `region`,
`termes_amazighs[]`, `elements_culturels[]`, `source`, `fiabilite`.

## Notes

- `text-embedding-004` → 768-dim vectors; this matches the Qdrant collection.
  Changing the embed model means recreating the collection (`--force`).
- MMR diversity (the anti-lissage core) is implemented in `app/rag.py`
  (`mmr_rerank`), because LlamaIndex's `vector_store_query_mode="mmr"` is a
  no-op with the Qdrant integration.
- To use a Qdrant server instead of embedded mode, change `get_client()` in
  `app/ingest.py` to `QdrantClient(url="http://localhost:6333")`.
