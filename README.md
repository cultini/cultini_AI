# cultini — Backend (short)

Developer-focused backend for CULTINI (RAG anti-lissage on Amazigh craft knowledge).

See the full developer documentation at [docs/README.md](docs/README.md).

Quickstart:

```bash
uv sync
cp .env.example .env
# set GEMINI_API_KEY in .env
uv run python -m app.ingest
uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs for the interactive API explorer.
