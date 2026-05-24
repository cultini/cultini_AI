"""The RAG query engine: retrieval + MMR diversity + cultural constitution.

Why the hand-rolled MMR: LlamaIndex's ``vector_store_query_mode="mmr"`` is a
no-op with the Qdrant integration (MMR re-ranking is only implemented for the
in-memory SimpleVectorStore). Since MMR diversity *is* our anti-lissage
mechanism, we over-fetch candidates and re-rank them ourselves with Maximal
Marginal Relevance, then synthesize over the diversified subset.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

import numpy as np
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.schema import NodeWithScore

from app import config, ingest, metrics

# --- The cultural constitution: the anti-lissage system prompt. ---
# Must contain {context_str} and {query_str}. Never use LlamaIndex's default.
CULTURAL_CONSTITUTION = PromptTemplate(
    "Tu es CULTINI, un assistant specialiste de l'artisanat amazigh (berbere).\n"
    "Tu reponds UNIQUEMENT a partir des sources fournies ci-dessous.\n\n"
    "CONSTITUTION CULTURELLE (regles imperatives) :\n"
    "1. N'utilise que les informations du contexte. Si l'information n'y est pas, "
    "dis-le clairement ; n'invente jamais.\n"
    "2. Conserve TOUS les termes amazighs tels quels (ex. talwt, tabzimt, azetta, yaz). "
    "Donne le terme amazigh puis explique-le ; ne le remplace pas par un equivalent francais.\n"
    "3. Mentionne toujours la region d'origine de ce que tu decris.\n"
    "4. Cite la source de chaque affirmation.\n"
    "5. Ne lisse jamais : si deux regions different, montre la difference au lieu d'une moyenne.\n"
    "6. Amplifie une culture existante, ne la genere pas.\n"
    "---------------------\n"
    "SOURCES :\n{context_str}\n"
    "---------------------\n"
    "Question : {query_str}\n"
    "Reponse (en francais, fidele et sourcee) :\n"
)


@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    """Load (or build once) the Qdrant-backed index, cached for the process."""
    return ingest.build_index(force=False)


@lru_cache(maxsize=1)
def get_referential() -> tuple[str, ...]:
    """Cultural-coverage referential, built once from the corpus."""
    return tuple(metrics.build_referential(ingest.load_fiches()))


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


def mmr_rerank(
    query_emb: list[float],
    cand_embs: list[list[float]],
    k: int,
    lambda_: float,
) -> list[int]:
    """Greedy Maximal Marginal Relevance. Returns indices into ``cand_embs``.

    Each step picks the candidate maximizing
        lambda * sim(query, cand) - (1 - lambda) * max sim(cand, already_picked).
    """
    if not cand_embs:
        return []
    q = np.asarray(query_emb, dtype=np.float32)[None, :]
    c = np.asarray(cand_embs, dtype=np.float32)
    sim_to_query = _cosine_matrix(q, c)[0]      # (n,)
    sim_between = _cosine_matrix(c, c)           # (n, n)

    selected: list[int] = []
    remaining = list(range(len(cand_embs)))
    k = min(k, len(cand_embs))
    while len(selected) < k:
        if not selected:
            best = int(np.argmax(sim_to_query[remaining]))
            selected.append(remaining.pop(best))
            continue
        best_idx, best_score = None, -np.inf
        for pos, idx in enumerate(remaining):
            redundancy = max(sim_between[idx][j] for j in selected)
            score = lambda_ * sim_to_query[idx] - (1 - lambda_) * redundancy
            if score > best_score:
                best_score, best_idx = score, pos
        selected.append(remaining.pop(best_idx))
    return selected


def _source_payload(node: NodeWithScore) -> dict:
    md = node.node.metadata
    return {
        "id": md.get("fiche_id", node.node.node_id),
        "titre": md.get("titre"),
        "region": md.get("region"),
        "source": md.get("source"),
        "categorie": md.get("categorie"),
        "fiabilite": md.get("fiabilite"),
        "score": round(float(node.score), 4) if node.score is not None else None,
    }


def _synthesize_cultural(
    retrieval_q: str, candidates: list[NodeWithScore]
) -> tuple[str, list[NodeWithScore]]:
    """MMR diversity re-rank ``candidates`` then synthesize with the constitution.

    ``retrieval_q`` is the query embedded for the MMR relevance term (it may carry
    conversation context). Returns ``(answer_text, selected_nodes)``.
    """
    query_emb = Settings.embed_model.get_query_embedding(retrieval_q)
    cand_texts = [c.node.get_content() for c in candidates]
    cand_embs = Settings.embed_model.get_text_embedding_batch(cand_texts)
    keep = mmr_rerank(query_emb, cand_embs, k=config.SIMILARITY_TOP_K, lambda_=config.MMR_LAMBDA)
    selected = [candidates[i] for i in keep]

    synthesizer = get_response_synthesizer(text_qa_template=CULTURAL_CONSTITUTION)
    response = synthesizer.synthesize(retrieval_q, nodes=selected)
    return str(response), selected


def _synthesize_cultural_stream(
    retrieval_q: str, candidates: list[NodeWithScore]
) -> tuple[Iterator[str], list[NodeWithScore]]:
    """Streaming twin of ``_synthesize_cultural``.

    Runs the same MMR selection synchronously (so the selected sources are known
    up front, for the ``meta`` event) but returns a *lazy* token generator that
    drives the LLM only as it is consumed. Returns ``(token_iter, selected)``.
    """
    query_emb = Settings.embed_model.get_query_embedding(retrieval_q)
    cand_texts = [c.node.get_content() for c in candidates]
    cand_embs = Settings.embed_model.get_text_embedding_batch(cand_texts)
    keep = mmr_rerank(query_emb, cand_embs, k=config.SIMILARITY_TOP_K, lambda_=config.MMR_LAMBDA)
    selected = [candidates[i] for i in keep]

    synthesizer = get_response_synthesizer(
        text_qa_template=CULTURAL_CONSTITUTION, streaming=True
    )
    streaming_response = synthesizer.synthesize(retrieval_q, nodes=selected)
    return streaming_response.response_gen, selected


def ask(question: str) -> dict:
    """Answer a question with diversified, sourced retrieval + the constitution.

    Returns ``{response, source_nodes, metrics}``.
    """
    config.configure_settings()
    index = get_index()

    # 1. Over-fetch candidates.
    retriever = index.as_retriever(similarity_top_k=config.OVERFETCH_K)
    candidates = retriever.retrieve(question)
    if not candidates:
        return {
            "response": "Aucune source pertinente trouvee dans le corpus pour cette question.",
            "source_nodes": [],
            "metrics": metrics.response_metrics("", list(get_referential())),
        }

    # 2. MMR diversity re-rank + synthesize with the constitution (anti-lissage).
    answer_text, selected = _synthesize_cultural(question, candidates)

    return {
        "response": answer_text,
        "source_nodes": [_source_payload(n) for n in selected],
        "metrics": metrics.response_metrics(answer_text, list(get_referential())),
    }


def ask_baseline(question: str) -> dict:
    """Baseline 'Gemini brut' : MEME modele, AUCUN retrieval, AUCUNE constitution.

    C'est le temoin de lissage — ce qu'un assistant generique repond sans
    ancrage culturel. La seule difference avec ask() est l'absence de RAG +
    constitution, donc l'ecart de metriques isole exactement l'apport de CULTINI.
    Renvoie ``{response, source_nodes, metrics}`` ; source_nodes est toujours [].
    """
    config.configure_settings()
    # Question nue, sans system prompt : c'est le "zero prompt" voulu.
    answer_text = str(Settings.llm.complete(question))
    return {
        "response": answer_text,
        "source_nodes": [],
        "metrics": metrics.response_metrics(answer_text, list(get_referential())),
    }


DIRECT_NOTICE = (
    "Reponse generee directement par le modele, sans source du corpus culturel CULTINI."
)


def _format_history(history: list[dict] | None) -> str:
    """Render recent turns as a compact text block (empty string when none)."""
    if not history:
        return ""
    lines = []
    for turn in history:
        speaker = "Utilisateur" if turn.get("role") == "user" else "CULTINI"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "Historique de la conversation :\n" + "\n".join(lines)


def ask_routed(question: str, history: list[dict] | None = None) -> dict:
    """Agentic router: gate between the cultural RAG path and the bare LLM.

    Retrieves first, then uses a retrieval-score gate — if the corpus has nothing
    relevant (no candidates, or top similarity below ``ROUTER_SCORE_THRESHOLD``),
    answer with the bare LLM and flag it via ``route='direct'`` + ``notice``.
    Otherwise run the full cultural path (``route='cultural'``).

    ``history`` (oldest-first ``{role, content}`` turns) is woven into the query
    used for retrieval and synthesis so follow-up questions resolve.

    Returns the ``ask()`` dict plus ``route``, ``router_score`` and ``notice``.
    """
    config.configure_settings()
    history_block = _format_history(history)
    contextual_q = f"{history_block}\n\nQuestion : {question}" if history_block else question

    index = get_index()
    retriever = index.as_retriever(similarity_top_k=config.OVERFETCH_K)

    # Gate on the BARE question so prior turns can't contaminate the routing
    # signal ("is this about our corpus?"). History is only woven back in once
    # we've decided to answer (for source retrieval + synthesis).
    gate_candidates = retriever.retrieve(question)
    top_score = max((c.score or 0.0) for c in gate_candidates) if gate_candidates else 0.0

    if not gate_candidates or top_score < config.ROUTER_SCORE_THRESHOLD:
        # Direct path: bare LLM, no retrieval, no constitution.
        answer_text = str(Settings.llm.complete(contextual_q))
        return {
            "response": answer_text,
            "source_nodes": [],
            "metrics": metrics.response_metrics(answer_text, list(get_referential())),
            "route": "direct",
            "router_score": round(top_score, 4),
            "notice": DIRECT_NOTICE,
        }

    # Cultural path: re-retrieve with context so follow-ups pull the right
    # sources, then MMR diversity + constitution.
    candidates = retriever.retrieve(contextual_q) if history_block else gate_candidates
    answer_text, selected = _synthesize_cultural(contextual_q, candidates)
    return {
        "response": answer_text,
        "source_nodes": [_source_payload(n) for n in selected],
        "metrics": metrics.response_metrics(answer_text, list(get_referential())),
        "route": "cultural",
        "router_score": round(top_score, 4),
        "notice": None,
    }


def ask_routed_stream(
    question: str, history: list[dict] | None = None
) -> tuple[dict, Iterator[str]]:
    """Streaming twin of ``ask_routed``.

    Returns ``(meta, token_iter)`` where ``meta`` is known synchronously
    (``{route, router_score, source_nodes}``) so the caller can emit it before
    any tokens, and ``token_iter`` yields text deltas lazily as the LLM
    generates. On the direct path the iterator first yields ``DIRECT_NOTICE``
    (with a blank line) so the "unsourced" disclaimer lands inline in the
    accumulated/persisted answer text.
    """
    config.configure_settings()
    history_block = _format_history(history)
    contextual_q = f"{history_block}\n\nQuestion : {question}" if history_block else question

    index = get_index()
    retriever = index.as_retriever(similarity_top_k=config.OVERFETCH_K)

    # Gate on the BARE question (same rationale as ask_routed).
    gate_candidates = retriever.retrieve(question)
    top_score = max((c.score or 0.0) for c in gate_candidates) if gate_candidates else 0.0

    if not gate_candidates or top_score < config.ROUTER_SCORE_THRESHOLD:
        meta = {
            "route": "direct",
            "router_score": round(top_score, 4),
            "source_nodes": [],
        }

        def _direct_gen() -> Iterator[str]:
            yield DIRECT_NOTICE + "\n\n"
            for chunk in Settings.llm.stream_complete(contextual_q):
                if chunk.delta:
                    yield chunk.delta

        return meta, _direct_gen()

    candidates = retriever.retrieve(contextual_q) if history_block else gate_candidates
    token_iter, selected = _synthesize_cultural_stream(contextual_q, candidates)
    meta = {
        "route": "cultural",
        "router_score": round(top_score, 4),
        "source_nodes": [_source_payload(n) for n in selected],
    }
    return meta, token_iter


if __name__ == "__main__":
    import json

    result = ask("Que signifie le losange dans le tissage amazigh ?")
    print(json.dumps(result, ensure_ascii=False, indent=2))
