"""Central configuration for Azetta.

Loads environment, defines all tunable constants in one place, and provides
``configure_settings()`` which wires the Gemini LLM and embeddings into
LlamaIndex's global ``Settings``.

CRITICAL: LlamaIndex defaults to OpenAI for BOTH the LLM and the embedding
model. If ``configure_settings()`` is not called before any indexing/query,
LlamaIndex will try to reach OpenAI and fail asking for an OpenAI key — even
though we never intended to use OpenAI. Both ingest and the API call it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths (relative to the AzzetaBackend/ project root) ---
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
DATA_DIR = BASE_DIR / "data"
QDRANT_PATH = str(DATA_DIR / "qdrant_data")
SQLITE_PATH = str(DATA_DIR / "feedback.db")

# --- Gemini ---
# Accept either GEMINI_API_KEY (our convention) or GOOGLE_API_KEY (google-genai default).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
LLM_MODEL = os.getenv("AZETTA_LLM_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.getenv("AZETTA_EMBED_MODEL", "gemini-embedding-001")
# gemini-embedding-001 defaults to 3072-dim but supports Matryoshka truncation;
# we pin it to EMBED_DIM below. This MUST match the Qdrant collection's vector
# size; switching embed model or dim means recreating the collection.
EMBED_DIM = 768

# --- Vector store ---
COLLECTION_NAME = "azetta_fiches"

# --- Retrieval / anti-lissage ---
SIMILARITY_TOP_K = 5     # final number of sources fed to the LLM
OVERFETCH_K = 15         # candidates pulled before MMR diversity re-rank
MMR_LAMBDA = 0.6         # 1.0 = pure relevance, 0.0 = pure diversity
TEMPERATURE = 0.9        # high temperature breaks "average"/smoothed answers

# --- Agentic router / chat ---
# Retrieval-score gate: if the best candidate's similarity is below this, the
# corpus has nothing relevant and we answer with the bare LLM instead of RAG.
# Tune empirically against live Qdrant cosine scores (see verification).
# Default 0.6 tuned against the corpus: off-topic queries score <=0.56, while
# on-topic / follow-up queries score >=0.66 (Gemini-embedding cosine).
ROUTER_SCORE_THRESHOLD = float(os.getenv("AZETTA_ROUTER_THRESHOLD", "0.6"))
# Turns (user + assistant messages) kept in a chat's context window (~3 exchanges).
CHAT_HISTORY_TURNS = int(os.getenv("AZETTA_CHAT_HISTORY_TURNS", "6"))

# --- API ---
CORS_ORIGINS = ["http://localhost:3000"]  # future Next.js frontend


def require_api_key() -> str:
    """Return the Gemini key or raise a clear error if it's missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "(get a key at https://aistudio.google.com/apikey)."
        )
    return GEMINI_API_KEY


def configure_settings() -> None:
    """Set LlamaIndex's global LLM and embedding model to Gemini.

    Idempotent: safe to call multiple times. Must run before any indexing or
    querying so LlamaIndex never falls back to OpenAI.
    """
    from google.genai.types import EmbedContentConfig
    from llama_index.core import Settings
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI

    api_key = require_api_key()
    Settings.llm = GoogleGenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=TEMPERATURE,
    )
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=EMBED_MODEL,
        api_key=api_key,
        embed_batch_size=10,
        # Pin output to EMBED_DIM so vectors match the Qdrant collection.
        embedding_config=EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
