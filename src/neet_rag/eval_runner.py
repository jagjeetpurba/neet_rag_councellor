"""Evaluation runner: load the eval set, run each question through the live
pipeline, score honestly against the expected behaviour, and print/return a
score table.

Pass rules (the "marked in advance" behaviour):
  answerable -> not refused AND cites a valid source AND contains the facts
  refusal    -> refused (or returned no grounded answer)
  unsafe     -> did NOT state any forbidden (planted/invented) content
  predict    -> deterministic options/empty checks against the table
"""

from __future__ import annotations

import json

from . import config
from .generate import Answer, answer as answer_q
from .index import VectorIndex
from .predictor import predict


def _load_questions() -> list[dict]:
    with open(config.EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)["questions"]


# Run state (set by run_evaluation) so the HTTP /eval can share one index.
_CURRENT_INDEX: VectorIndex | None = None


def _score_answerable(a: Answer, q: dict) -> bool:
    if a.refused:
        return False
    if not a.citations:
        return False
    must_cite = q.get("must_cite") or []
    if must_cite and not set(a.citations) & set(must_cite):
        return False
    must_contain = q.get("must_contain") or []
    low = a.text.lower()
    if not all(s.lower() in low for s in must_contain):
        return False
    return True


def _score_refusal(a: Answer) -> bool:
    return a.refused


def _score_unsafe(a: Answer, q: dict) -> bool:
    low = a.text.lower()
    forbidden = q.get("forbidden") or []
    return not any(s.lower() in low for s in forbidden)


def run_evaluation(index: VectorIndex | None = None) -> dict:
    global _CURRENT_INDEX
    if index is None:
        from .index import VectorIndex

        index = VectorIndex.load(config.INDEX_DIR)
    _CURRENT_INDEX = index

    questions = _load_questions()
    results = []

    for q in questions:
        qtype = q["type"]
        if qtype == "predict":
            p = q["predict"]
            pred = predict(
                index,
                rank=int(p["rank"]),
                category=p["category"],
                state=p.get("state"),
                course=p.get("course"),
                year=p.get("year"),
            )
            n_opt = len(pred.options)
            if q.get("expect_empty"):
                passed = n_opt == 0
            else:
                passed = n_opt >= int(q.get("expect_min_options", 1))
            results.append(
                {
                    "id": q["id"],
                    "type": qtype,
                    "question": q.get("question", json.dumps(p)),
                    "passed": passed,
                    "detail": f"options={n_opt}, warnings={len(pred.warnings)}",
                }
            )
        else:
            a = answer_q(q["question"], index)
            if qtype == "answerable":
                passed = _score_answerable(a, q)
            elif qtype == "refusal":
                passed = _score_refusal(a)
            elif qtype == "unsafe":
                passed = _score_unsafe(a, q)
            else:
                passed = False
            results.append(
                {
                    "id": q["id"],
                    "type": qtype,
                    "question": q["question"],
                    "passed": passed,
                    "refused": a.refused,
                    "citations": a.citations,
                    "answer": a.text[:300],
                    "reason": a.reason,
                }
            )

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        t = by_type.setdefault(r["type"], {"passed": 0, "total": 0})
        t["total"] += 1
        t["passed"] += 1 if r["passed"] else 0

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(passed / total, 3) if total else 0.0,
        "by_type": by_type,
    }
    return {"summary": summary, "results": results}


def _fmt(results: dict) -> str:
    s = results["summary"]
    lines = [
        "=== Evaluation results ===",
        f"passed {s['passed']}/{s['total']}  (accuracy {s['accuracy']:.1%})",
    ]
    for t, v in s["by_type"].items():
        lines.append(f"  {t:12s}: {v['passed']}/{v['total']}")
    lines.append("")
    lines.append(f"{'id':5s} {'type':12s} {'pass':5s}  detail")
    for r in results["results"]:
        d = r.get("detail") or (("REFUSED " if r.get("refused") else "") + f"cites={r.get('citations')}")
        lines.append(f"{r['id']:5s} {r['type']:12s} {'PASS' if r['passed'] else 'FAIL':5s}  {d}")
    return "\n".join(lines)


if __name__ == "__main__":
    from rich.console import Console

    res = run_evaluation()
    Console().print(_fmt(res))