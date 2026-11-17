"""Track A: NEET college predictor.

Deterministic filtering over the ingested table (never the LLM guessing a
number): given a rank + category (+ optional state/course/year), list
"eligible" options (your rank is at least as good as the published closing
rank) and "reach" options (the cut-off was better than your rank, so
unlikely). Every option carries its source id, so the answer is checkable.

This is deliberately separate from the semantic retrieval path: structured
predicates (rank <= closing_rank, category == X) are exact and must not be
left to a generative model, which is the whole point of the "unsafe advice"
test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .index import Passage, VectorIndex


@dataclass
class Option:
    college: str
    course: str
    quota: str
    state: str
    category: str
    year: int
    opening_rank: int
    closing_rank: int
    seats: int
    source_id: str
    kind: str = "eligible"  # "eligible" | "reach"


@dataclass
class Prediction:
    rank: int
    category: str
    state: str | None
    course: str | None
    year: int | None
    options: list[Option] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _meta_frame(index: VectorIndex) -> pd.DataFrame:
    rows = []
    for p in index.passages:
        m = dict(p.meta)
        m["source_id"] = p.source_id
        rows.append(m)
    return pd.DataFrame(rows)


def predict(
    index: VectorIndex,
    rank: int,
    category: str,
    state: str | None = None,
    course: str | None = None,
    year: int | None = None,
    quota: str | None = None,
    reach_band: float = 0.20,
    max_reach: int = 5,
) -> Prediction:
    df = _meta_frame(index)

    if df.empty:
        return Prediction(rank, category, state, course, year, warnings=["index is empty"])

    # Validate category against what is in the data -> clear "no data" warning.
    known_cats = set(df["category"].unique())
    pred = Prediction(rank=rank, category=category, state=state, course=course, year=year)
    if category not in known_cats:
        pred.warnings.append(
            f"category '{category}' not present in sources (known: {sorted(known_cats)})"
        )
        return pred

    # latest year defaults to the most recent in the data.
    eff_year = year if year is not None else int(df["year"].max())

    mask = (df["category"] == category) & (df["year"] == eff_year)
    if course is not None:
        mask &= df["course"] == course
    if quota is not None:
        # quota (AIQ / State) is matched exactly.
        mask &= df["quota"].str.lower() == quota.lower()
    if state is not None:
        # a state query matches the college's location state (case-insensitive).
        mask &= df["state"].str.lower() == state.lower()

    sub = df[mask]
    if sub.empty:
        pred.warnings.append(
            f"no cut-off data for category={category}, year={eff_year}"
            + (f", state={state}" if state else "")
            + (f", quota={quota}" if quota else "")
            + (f", course={course}" if course else "")
        )
        return pred

    eligible = sub[sub["closing_rank"] >= rank].sort_values("closing_rank")
    reach = sub[sub["closing_rank"] < rank].sort_values("closing_rank", ascending=False)

    # cap reach to a band around the rank + a count cap, to avoid listing noise.
    lo = int(rank * (1 - reach_band))
    reach = reach[reach["closing_rank"] >= lo].head(max_reach)

    def _opt(r: pd.Series, kind: str) -> Option:
        return Option(
            college=str(r["college"]),
            course=str(r["course"]),
            quota=str(r["quota"]),
            state=str(r["state"]),
            category=str(r["category"]),
            year=int(r["year"]),
            opening_rank=int(r["opening_rank"]),
            closing_rank=int(r["closing_rank"]),
            seats=int(r["seats"]),
            source_id=str(r["source_id"]),
            kind=kind,
        )

    pred.options = [_opt(r, "eligible") for _, r in eligible.iterrows()]
    pred.options += [_opt(r, "reach") for _, r in reach.iterrows()]
    return pred