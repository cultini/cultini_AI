"""Corpus ingestion: JSON fiches -> LlamaIndex TextNodes -> Qdrant.

Design choices:
- One fiche = one node. We build ``TextNode``s directly (no chunking) so each
  source stays atomic and citable, and we set ``id_`` to the fiche id so
  re-ingesting upserts instead of duplicating.
- Persistent + load-or-build. The embedded Qdrant store survives between runs;
  we only re-embed when the collection is missing/empty or ``force=True``.

Run as a CLI:  ``uv run python -m app.ingest [--force]``
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app import config

REQUIRED_FIELDS = {
    "id", "categorie", "titre", "contenu", "region",
    "termes_amazighs", "elements_culturels", "source", "fiabilite",
}

# Qdrant point IDs must be UUIDs or unsigned ints, but our fiche ids are
# human-readable slugs (e.g. "bijou_corail_argent"). Derive a deterministic
# UUID from the slug so re-ingesting upserts the same point, and keep the slug
# in metadata as ``fiche_id`` for citation.
_FICHE_NAMESPACE = uuid.UUID("a3b1c0de-0000-4000-8000-617a65747461")  # "azetta"


def fiche_uuid(fiche_id: str) -> str:
    return str(uuid.uuid5(_FICHE_NAMESPACE, fiche_id))


def load_fiches(corpus_dir: Path | None = None) -> list[dict]:
    """Read and validate every ``*.json`` fiche in the corpus directory."""
    corpus_dir = corpus_dir or config.CORPUS_DIR
    fiches: list[dict] = []
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED_FIELDS - data.keys()
        if missing:
            raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
        fiches.append(data)
    if not fiches:
        raise RuntimeError(f"No fiches found in {corpus_dir}")
    return fiches


def build_nodes(fiches: list[dict]) -> list[TextNode]:
    """Turn fiches into one TextNode each, carrying metadata for citation/filtering."""
    nodes: list[TextNode] = []
    for f in fiches:
        nodes.append(
            TextNode(
                id_=fiche_uuid(f["id"]),
                text=f"{f['titre']}\n{f['contenu']}",
                metadata={
                    "fiche_id": f["id"],
                    "titre": f["titre"],
                    "region": f["region"],
                    "source": f["source"],
                    "categorie": f["categorie"],
                    "termes_amazighs": f["termes_amazighs"],
                    "elements_culturels": f["elements_culturels"],
                    "fiabilite": f["fiabilite"],
                },
                # Keep cultural metadata out of the embedded/LLM text so it
                # doesn't pollute similarity or the answer, but stays available.
                excluded_embed_metadata_keys=[
                    "fiche_id", "source", "fiabilite", "termes_amazighs", "elements_culturels",
                ],
                excluded_llm_metadata_keys=[
                    "fiche_id", "termes_amazighs", "elements_culturels", "fiabilite",
                ],
            )
        )
    return nodes


def get_client() -> QdrantClient:
    """Single place that opens the Qdrant client (embedded, on-disk).

    NOTE: embedded mode takes an exclusive lock on the data dir — run a single
    process (no ``uvicorn --reload``). To switch to a Docker server, replace
    with ``QdrantClient(url="http://localhost:6333")``.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=config.QDRANT_PATH)


def _vector_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(client=client, collection_name=config.COLLECTION_NAME)


def _collection_ready(client: QdrantClient) -> bool:
    if not client.collection_exists(config.COLLECTION_NAME):
        return False
    return client.count(config.COLLECTION_NAME, exact=True).count > 0


def build_index(force: bool = False) -> VectorStoreIndex:
    """Load the index from Qdrant, or build (embed + write) it if needed."""
    config.configure_settings()
    client = get_client()

    if force and client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)

    vector_store = _vector_store(client)

    if not force and _collection_ready(client):
        # Already indexed — load without re-embedding.
        return VectorStoreIndex.from_vector_store(vector_store)

    fiches = load_fiches()
    nodes = build_nodes(fiches)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print(f"Indexed {len(nodes)} fiches into Qdrant collection '{config.COLLECTION_NAME}'.")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Azetta corpus into Qdrant.")
    parser.add_argument("--force", action="store_true", help="Drop and rebuild the collection.")
    args = parser.parse_args()

    build_index(force=args.force)
    client = get_client()
    count = client.count(config.COLLECTION_NAME, exact=True).count
    print(f"Collection '{config.COLLECTION_NAME}' now holds {count} vectors.")


if __name__ == "__main__":
    main()
