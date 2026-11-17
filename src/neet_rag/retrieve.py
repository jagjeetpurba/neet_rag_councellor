"""Retrieval: embed the query, top-k cosine search, with a refusal gate.

The refusal gate fires BEFORE the LLM when nothing retrieved is relevant
enough, which is the single most reliable way to stop hallucinated answers
("cut-off for a college not in the index" -> no relevant passage -> refuse).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .embed import embed_one
from .index import Passage, VectorIndex


@dataclass
class RetrievalResult:
    hits: list[tuple[Passage, float]]
    refused: bool
    reason: str = ""


# Function/grammar and general domain words that appear capitalised in
# questions but are NOT entities. Excluded from the coverage guard so a normal
# "What is the NEET cut-off ..." question never triggers a false refusal.
_FUNCTION_WORDS = {
    "what", "which", "how", "when", "who", "why", "where", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "should", "would", "will",
    "may", "might", "shall", "the", "a", "an", "of", "for", "in", "on", "at",
    "to", "with", "my", "me", "i", "if", "am", "tell", "list", "show", "give",
    "and", "or", "not", "no", "yes", "please", "rank", "year", "category",
    "neet", "ug", "pg", "india", "all", "quota", "aiq", "general", "admission",
    "counselling", "counseling", "college", "colleges", "medical", "government",
    "seats", "closing", "opening", "cutoff", "every", "this", "that", "any",
    "some", "know", "want", "need", "state", "open", "obc", "sc", "st", "ews",
}

# The corpus is strictly cut-off / seat-matrix data, so a question asking for
# one of these — attributes we do not store — is out of scope by construction
# and can be refused deterministically.
_OUT_OF_SCOPE_WORDS = {
    "fee", "fees", "tuition", "hostel", "stipend", "salary", "address",
    "phone", "contact", "email", "eligibility", "syllabus", "brochure",
    "admit", "pattern", "ranking", "placement", "coaching", "institute",
    "mock", "preparation", "career", "salary", "stipend", "donation", "capitation",
}


def _known_words(index: VectorIndex) -> set[str]:
    """Every word appearing in an entity field (college, state, course,
    category, quota) of the index — the 'known universe' of the corpus."""
    words: set[str] = set()
    for p in index.passages:
        for key in ("college", "state", "course", "category", "quota"):
            for w in str(p.meta.get(key, "")).split():
                words.add(w.lower())
    return words


def _known_years(index: VectorIndex) -> set[str]:
    return {str(p.meta.get("year")) for p in index.passages}


def _unsupported_entity(question: str, index: VectorIndex) -> str | None:
    """Deterministic refusal backstop, three signals (any one refuses):

    1. a named year that is not in the corpus;
    2. a named proper noun that is not in the corpus (e.g. 'AIIMS Jodhpur');
    3. an out-of-scope attribute word (fee, hostel, eligibility, coaching ...).
    """
    # (1) year not in sources
    for y in set(re.findall(r"\b(?:19|20)\d{2}\b", question)):
        if y not in _known_years(index):
            return f"year {y} not in sources"

    low = question.lower()

    # (3) out-of-scope attribute
    for w in _OUT_OF_SCOPE_WORDS:
        if re.search(rf"\b{w}\b", low):
            return f"'{w}' not in sources"

    # (2) proper noun not in corpus
    known = _known_words(index) | _FUNCTION_WORDS
    for token in re.findall(r"\b[A-Z][a-zA-Z]+\b", question):
        if token.lower() not in known:
            return f"'{token}' not in sources"
    return None


def retrieve(query: str, index: VectorIndex, top_k: int | None = None) -> RetrievalResult:
    top_k = top_k or config.TOP_K
    qvec = embed_one(query)
    hits = index.search(qvec, top_k=top_k)

    # (1) entity-coverage guard: named-but-absent year/entity -> refuse.
    entity_reason = _unsupported_entity(query, index)
    if entity_reason:
        return RetrievalResult(hits=hits, refused=True, reason=entity_reason)

    # (2) similarity gate: nothing relevant retrieved -> refuse.
    best_sim = hits[0][1] if hits else 0.0
    if not hits or best_sim < config.MIN_SIMILARITY:
        return RetrievalResult(
            hits=hits,
            refused=True,
            reason=f"best similarity {best_sim:.2f} below threshold {config.MIN_SIMILARITY}",
        )
    return RetrievalResult(hits=hits, refused=False)