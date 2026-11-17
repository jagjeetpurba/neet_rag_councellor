"""Generation: prompt the local model with retrieved context, enforce
citations and refusal, and verify that every citation points to a real
retrieved source (post-generation citation audit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import ollama

from . import config
from .index import Passage
from .retrieve import RetrievalResult, retrieve

REFUSAL_TOKEN = "NOT_IN_SOURCES"

SYSTEM_PROMPT = (
    "You are a NEET UG counselling assistant. You may answer ONLY from the "
    "retrieved source passages given to you. Every factual claim MUST cite the "
    "source id(s) it comes from, formatted like [neet_cutoffs#17]. "
    "If the retrieved context does not contain the information needed to answer, "
    f'reply with exactly: {REFUSAL_TOKEN}. '
    "Never invent a cut-off, rank, seat count, or fact. Do not give personal "
    "medical or career guarantees; you provide data-backed options and their limits."
)


@dataclass
class Answer:
    text: str
    refused: bool
    citations: list[str]
    reason: str = ""


def _build_context(hits: list[tuple[Passage, float]]) -> str:
    lines = []
    for i, (p, sim) in enumerate(hits, start=1):
        lines.append(f"[{i}] {p.text} (source: {p.source_id})")
    return "\n".join(lines)


def _call_llm(context: str, question: str) -> str:
    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.chat(
        model=config.GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer with citations. If unsupported, reply "
                    f"{REFUSAL_TOKEN}."
                ),
            },
        ],
        options={"temperature": config.TEMPERATURE, "num_predict": config.MAX_TOKENS},
    )
    return resp["message"]["content"].strip()


def _extract_citations(text: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    """Return (cleaned_text, valid_citations). Drops any [id] that is not in the
    retrieved set so the answer never cites a hallucinated id."""
    seen: list[str] = []

    def _repl(m: re.Match[str]) -> str:
        cid = m.group(1)
        if cid in valid_ids:
            if cid not in seen:
                seen.append(cid)
            return f"[{cid}]"
        return "[source-removed]"

    cleaned = re.sub(r"\[([A-Za-z0-9_#-]+)\]", _repl, text)
    return cleaned, seen


def ensure_generator_model() -> None:
    from .embed import _installed_models

    client = ollama.Client(host=config.OLLAMA_HOST)
    installed = _installed_models(client)
    base = config.GENERATOR_MODEL.split(":")[0]
    if base not in installed:
        print(f"[generate] pulling {config.GENERATOR_MODEL} ...")
        client.pull(config.GENERATOR_MODEL)


def answer(question: str, index, ret: RetrievalResult | None = None) -> Answer:
    """Full pipeline step: retrieve (unless provided), generate, audit."""
    if ret is None:
        ret = retrieve(question, index)

    if ret.refused:
        return Answer(
            text="Not in the sources. I could not find information that supports an answer.",
            refused=True,
            citations=[],
            reason=ret.reason,
        )

    context = _build_context(ret.hits)
    raw = _call_llm(context, question)

    if REFUSAL_TOKEN in raw:
        return Answer(
            text="Not in the sources. The retrieved material does not support an answer.",
            refused=True,
            citations=[],
            reason=f"model returned {REFUSAL_TOKEN}",
        )

    valid_ids = {p.source_id for p, _ in ret.hits}
    cleaned, citations = _extract_citations(raw, valid_ids)

    # Hard rule: an answer with zero valid citations on an answerable question
    # is treated as ungrounded -> refuse rather than hand over free text.
    if not citations:
        return Answer(
            text="Not in the sources. The model could not ground its answer in the retrieved passages.",
            refused=True,
            citations=[],
            reason="no valid citations produced",
        )

    return Answer(text=cleaned, refused=False, citations=citations)