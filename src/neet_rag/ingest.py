"""Ingest: load a table (CSV) -> build cited passages -> embed -> persist index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .embed import embed_texts, ensure_embed_model
from .index import Passage, VectorIndex


def row_to_passage(row: pd.Series) -> Passage:
    """Turn one table row into a natural-language passage with a stable id."""
    fields = [
        row["college"],
        f"{row['course']}",
        f"{row['quota']} quota",
        f"{row['state']}",
        f"{row['category']} category",
        f"{row['year']}",
        f"opening rank {int(row['opening_rank'])}",
        f"closing rank {int(row['closing_rank'])}",
        f"{int(row['seats'])} seats",
    ]
    text = ", ".join(fields) + "."
    return Passage(
        source_id=str(row["source_id"]),
        text=text,
        meta={
            "college": str(row["college"]),
            "course": str(row["course"]),
            "quota": str(row["quota"]),
            "state": str(row["state"]),
            "category": str(row["category"]),
            "year": int(row["year"]),
            "opening_rank": int(row["opening_rank"]),
            "closing_rank": int(row["closing_rank"]),
            "seats": int(row["seats"]),
        },
    )


def build_index(data_dir: Path | None = None, force: bool = False) -> VectorIndex:
    """Build (or reuse) the vector index from every CSV in the data dir."""
    data_dir = Path(data_dir or config.ACTIVE_DATA_DIR)
    out_dir = config.INDEX_DIR
    if out_dir.exists() and (out_dir / "embeddings.npz").exists() and not force:
        print(f"[ingest] index already exists at {out_dir}; skipping (use --force to rebuild)")
        return VectorIndex.load(out_dir)

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSV files found in {data_dir}")

    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"[ingest] loaded {len(df)} rows from {len(csvs)} file(s) in {data_dir}")

    passages = [row_to_passage(r) for _, r in df.iterrows()]
    ensure_embed_model()
    print(f"[ingest] embedding {len(passages)} passages with {config.EMBED_MODEL} ...")
    embeddings = embed_texts([p.text for p in passages])

    index = VectorIndex(passages, embeddings)
    index.save(out_dir)
    print(f"[ingest] built + saved index: {index.size} passages -> {out_dir}")
    return index