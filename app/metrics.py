"""Anti-lissage metrics — the home-grown numbers that LlamaIndex does not give us.

- Cultural Coverage: of the whole corpus's cultural vocabulary (the union of
  every fiche's ``elements_culturels`` + ``termes_amazighs``), what fraction
  does a given answer actually mobilise? This is the headline metric: ask the
  same question to a generic LLM and to Azetta, measure both, compare.
- Distinct-n: ratio of unique n-grams — lexical richness / anti-repetition.
- Self-BLEU proxy: average pairwise n-gram overlap across several answers to the
  SAME prompt. Low = diverse generations = anti-lissage. We use a Jaccard-on-
  n-grams proxy to avoid nltk's BLEU smoothing pitfalls and any data download.

A deliberately dependency-light regex tokenizer is used (no nltk ``punkt``
download at runtime).
"""

from __future__ import annotations

import re
import unicodedata
from itertools import combinations

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    """Lowercase + strip accents, for accent-insensitive matching."""
    return _strip_accents(text.lower())


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_referential(fiches: list[dict]) -> list[str]:
    """Union of all cultural elements + Tamazight terms across the corpus.

    Returned as a sorted list of normalized terms (deduplicated).
    """
    terms: set[str] = set()
    for f in fiches:
        for item in list(f.get("elements_culturels", [])) + list(f.get("termes_amazighs", [])):
            t = normalize(item).strip()
            if t:
                terms.add(t)
    return sorted(terms)


def cultural_coverage(response_text: str, referential: list[str]) -> dict:
    """Fraction of referential terms that appear in the response.

    Returns the percentage plus the matched/total counts and the matched terms,
    so the UI can show *which* cultural elements were mobilised.
    """
    if not referential:
        return {"percent": 0.0, "matched": 0, "total": 0, "matched_terms": []}
    norm_resp = normalize(response_text)
    matched = [t for t in referential if t in norm_resp]
    return {
        "percent": round(100.0 * len(matched) / len(referential), 1),
        "matched": len(matched),
        "total": len(referential),
        "matched_terms": matched,
    }


def distinct_n(text: str, n: int = 2) -> float:
    """Unique n-grams / total n-grams in [0, 1]. Higher = less repetitive."""
    grams = _ngrams(tokenize(text), n)
    if not grams:
        return 0.0
    return round(len(set(grams)) / len(grams), 3)


def _jaccard_ngrams(a: str, b: str, n: int) -> float:
    ga, gb = set(_ngrams(tokenize(a), n)), set(_ngrams(tokenize(b), n))
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def self_bleu_proxy(responses: list[str], n: int = 2) -> float:
    """Average pairwise n-gram Jaccard overlap across answers to the same prompt.

    0.0 = totally diverse generations, 1.0 = identical. Needs >= 2 responses.
    """
    if len(responses) < 2:
        return 0.0
    overlaps = [_jaccard_ngrams(a, b, n) for a, b in combinations(responses, 2)]
    return round(sum(overlaps) / len(overlaps), 3)


def response_metrics(response_text: str, referential: list[str]) -> dict:
    """Per-answer metric bundle attached to every /ask response."""
    return {
        "cultural_coverage": cultural_coverage(response_text, referential),
        "distinct_1": distinct_n(response_text, 1),
        "distinct_2": distinct_n(response_text, 2),
    }
