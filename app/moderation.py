"""Auto-filter for incoming contributions (the "filtre auto" in the flow).

A submission is screened on three axes before it ever reaches a human:

  • spam       — pure heuristics (too short, link farms, char repetition).
  • doublon    — embedding similarity against the live corpus via the RAG index.
  • contenu IA — an LLM verdict on whether the prose reads as machine-generated.

Design principle: **fail open**. The duplicate and AI checks depend on Gemini
(embeddings + LLM) and on the Qdrant index being loaded. If either is missing
(no API key, index not built, network error) the check returns "not flagged"
rather than raising, so a legitimate contribution is never lost to an
infrastructure hiccup — it just lands in the moderation queue for a human.
``screen()`` returns a flags dict; the caller decides what to auto-reject.
"""

from __future__ import annotations

import json
import re

from app import config

# A submission tripping spam OR doublon is auto-rejected; ``ai_generated`` is a
# warning surfaced to the expert (LLM AI-detection is too unreliable to reject on
# its own). See ``decide_status`` for the policy.
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_REPEAT_RE = re.compile(r"(.)\1{6,}")  # same char 7+ times in a row


def check_spam(titre: str, contenu: str) -> tuple[bool, str | None]:
    """Cheap, dependency-free spam heuristics. Returns ``(is_spam, reason)``."""
    body = contenu.strip()
    if len(body) < config.SPAM_MIN_CONTENT_LEN:
        return True, f"Contenu trop court (< {config.SPAM_MIN_CONTENT_LEN} caractères)."
    if len(_URL_RE.findall(f"{titre} {body}")) > config.SPAM_MAX_LINK_COUNT:
        return True, "Trop de liens — ressemble à du spam."
    if _REPEAT_RE.search(body):
        return True, "Répétition anormale de caractères."
    letters = sum(c.isalpha() for c in body)
    if letters / max(len(body), 1) < 0.5:
        return True, "Trop peu de texte alphabétique — ressemble à du spam."
    return False, None


def check_duplicate(titre: str, contenu: str) -> tuple[bool, dict | None]:
    """Embedding-similarity duplicate check against the live corpus.

    Embeds ``"{titre}\\n{contenu}"`` (the same shape nodes are built from) and
    retrieves the single closest fiche. Flags a duplicate when the cosine score
    is at or above ``DUPLICATE_SCORE_THRESHOLD``. Fails open on any error.
    """
    try:
        from app import rag

        index = rag.get_index()
        retriever = index.as_retriever(similarity_top_k=1)
        hits = retriever.retrieve(f"{titre}\n{contenu}")
        if not hits:
            return False, None
        top = hits[0]
        score = float(top.score or 0.0)
        if score >= config.DUPLICATE_SCORE_THRESHOLD:
            md = top.node.metadata
            return True, {
                "duplicate_of": md.get("fiche_id"),
                "duplicate_titre": md.get("titre"),
                "score": round(score, 4),
            }
        return False, None
    except Exception:  # index missing / no embeddings / network — don't block.
        return False, None


_AI_CLASSIFIER_PROMPT = (
    "Tu es un détecteur de texte généré par IA. On te donne une contribution "
    "destinée à un corpus sur l'artisanat amazigh. Réponds UNIQUEMENT par un "
    'objet JSON {"ai_generated": true|false, "reason": "courte explication"}. '
    "Indices d'un texte IA : ton lisse et générique, absence de détails concrets, "
    "tournures encyclopédiques sans source vécue.\n\n"
    "Titre : {titre}\nContenu : {contenu}\n\nJSON :"
)


def check_ai_generated(titre: str, contenu: str) -> tuple[bool, str | None]:
    """Ask the LLM whether the text reads as machine-generated. Fails open."""
    try:
        from llama_index.core import Settings

        config.configure_settings()
        prompt = _AI_CLASSIFIER_PROMPT.format(titre=titre, contenu=contenu)
        raw = str(Settings.llm.complete(prompt)).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return False, None
        verdict = json.loads(match.group(0))
        return bool(verdict.get("ai_generated")), verdict.get("reason")
    except Exception:  # no key / parse failure — treat as human, let a human judge.
        return False, None


def screen(titre: str, contenu: str) -> dict:
    """Run all three checks and return a flags dict.

    Shape::

        {
          "spam": bool, "spam_reason": str | None,
          "duplicate": bool, "duplicate_info": dict | None,
          "ai_generated": bool, "ai_reason": str | None,
        }
    """
    is_spam, spam_reason = check_spam(titre, contenu)
    # Skip the costly duplicate/AI checks once it's already spam.
    if is_spam:
        return {
            "spam": True, "spam_reason": spam_reason,
            "duplicate": False, "duplicate_info": None,
            "ai_generated": False, "ai_reason": None,
        }
    is_dup, dup_info = check_duplicate(titre, contenu)
    is_ai, ai_reason = check_ai_generated(titre, contenu)
    return {
        "spam": False, "spam_reason": None,
        "duplicate": is_dup, "duplicate_info": dup_info,
        "ai_generated": is_ai, "ai_reason": ai_reason,
    }


def decide_status(flags: dict) -> tuple[str, str]:
    """Map flags to a moderation status + a user-facing French message.

    ``spam`` and ``doublon`` are auto-rejected (high confidence); an
    ``ai_generated`` flag alone still goes to the human queue as a warning.
    """
    if flags.get("spam"):
        return "auto_rejected", flags.get("spam_reason") or "Contribution filtrée (spam)."
    if flags.get("duplicate"):
        info = flags.get("duplicate_info") or {}
        titre = info.get("duplicate_titre")
        suffix = f" (proche de « {titre} »)" if titre else ""
        return "auto_rejected", f"Une fiche très similaire existe déjà{suffix}."
    return "pending", "Merci ! Votre contribution est en attente de validation par un expert."
