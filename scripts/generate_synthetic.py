"""Generate a synthetic but realistic NEET UG (MBBS/BDS) cut-off table.

This is a **seed** dataset used to develop and demo the pipeline end-to-end
while (and in case) real public data is still being cleaned. It is fully
synthetic, deterministic, and released as CC0 (public domain): no real
student, college, or government figures are used.

Run:  python scripts/generate_synthetic.py
Output: data/synthetic/neet_cutoffs.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "neet_cutoffs.csv"

# AIQ = All-India Quota (15%), State = state quota (rest). NEET UG categories.
COLLEGES = {
    "AIIMS New Delhi": ("Delhi", "AIQ"),
    "AIIMS Kalyani": ("West Bengal", "AIQ"),
    "Medical College Kolkata": ("West Bengal", "State"),
    "IPGMER Kolkata": ("West Bengal", "State"),
    "NRS Medical College": ("West Bengal", "State"),
    "Maulana Azad Medical College": ("Delhi", "AIQ"),
    "Grant Medical College Mumbai": ("Maharashtra", "State"),
    "King George's Medical University": ("Uttar Pradesh", "State"),
    "JIPMER Puducherry": ("Puducherry", "AIQ"),
    "Government Medical College Cuttack": ("Odisha", "State"),
}

COURSES = ["MBBS", "BDS"]
CATEGORIES = ["Open", "EWS", "OBC", "SC", "ST"]
YEARS = [2023, 2024]

# Base closing rank by college "tier" (lower = more competitive). Synthetic.
_BASE = {
    "AIIMS New Delhi": 50,
    "JIPMER Puducherry": 900,
    "Maulana Azad Medical College": 1200,
    "Grant Medical College Mumbai": 1800,
    "IPGMER Kolkata": 2600,
    "Medical College Kolkata": 5200,
    "King George's Medical University": 6400,
    "NRS Medical College": 9000,
    "AIIMS Kalyani": 10500,
    "Government Medical College Cuttack": 14000,
}

# Category multiplier: closing rank widens for reserved categories (synthetic).
_CAT_MULT = {"Open": 1.0, "EWS": 1.4, "OBC": 1.7, "SC": 2.6, "ST": 3.4}
_COURSE_MULT = {"MBBS": 1.0, "BDS": 2.5}  # BDS less competitive


def build_rows(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for college, (state, quota) in COLLEGES.items():
        base = _BASE[college]
        for course in COURSES:
            for category in CATEGORIES:
                for year in YEARS:
                    closing = int(
                        base * _CAT_MULT[category] * _COURSE_MULT[course] * (1.0 + 0.03 * (2024 - year))
                    )
                    # opening rank is earlier than closing; spread is realistic.
                    opening = max(1, int(closing * rng.uniform(0.72, 0.82)))
                    # synthetic AO/seat count, deterministic-ish.
                    seats = rng.choice([15, 30, 50, 85, 120])
                    rows.append(
                        {
                            "source_id": "",  # filled after sort, below
                            "college": college,
                            "course": course,
                            "quota": quota,
                            "state": state,
                            "category": category,
                            "year": year,
                            "opening_rank": opening,
                            "closing_rank": closing,
                            "seats": seats,
                        }
                    )
    # stable ordering and stable source ids: row number -> neet_cutoffs#N
    rows.sort(key=lambda r: (r["college"], r["course"], r["category"], r["year"]))
    for i, r in enumerate(rows, start=1):
        r["source_id"] = f"neet_cutoffs#{i}"
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_rows())
    # source_id first so it reads naturally.
    cols = ["source_id"] + [c for c in df.columns if c != "source_id"]
    df = df[cols]
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()