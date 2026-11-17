"""FastAPI service exposing the RAG Q&A and the Track A predictor.

Endpoints:
  GET  /health    -> liveness + hardware/model summary
  POST /ask       -> {"question": str} -> cited answer or refusal
  POST /predict   -> rank/category/state/course/year -> options with sources
  POST /ingest    -> rebuild the index from the active data dir
  POST /eval      -> run the eval set, return the score table
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config
from .generate import answer as answer_question
from .index import VectorIndex
from .ingest import build_index
from .predictor import predict

app = FastAPI(title="NEET RAG Counsellor", version="0.1.0")

_index: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        if not (config.INDEX_DIR / "embeddings.npz").exists():
            build_index()
        _index = VectorIndex.load(config.INDEX_DIR)
    return _index


class AskReq(BaseModel):
    question: str


class AskResp(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[str]
    reason: str = ""


class PredictReq(BaseModel):
    rank: int
    category: str
    state: str | None = None
    course: str | None = None
    year: int | None = None
    quota: str | None = None


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    ui_file = config.PROJECT_ROOT / "web" / "index.html"
    return ui_file.read_text(encoding="utf-8") if ui_file.exists() else "UI not found"


@app.get("/health")
def health() -> dict:
    idx = get_index()
    return {
        "status": "ok",
        "model": config.GENERATOR_MODEL,
        "embed_model": config.EMBED_MODEL,
        "index_size": idx.size,
        "assumed_gpu": config.ASSUMED_GPU,
        "actual_runtime": config.ACTUAL_RUNTIME,
    }


@app.post("/ask", response_model=AskResp)
def ask(req: AskReq) -> AskResp:
    a = answer_question(req.question, get_index())
    return AskResp(
        question=req.question,
        answer=a.text,
        refused=a.refused,
        citations=a.citations,
        reason=a.reason,
    )


@app.post("/predict")
def predict_endpoint(req: PredictReq) -> dict:
    idx = get_index()
    pred = predict(
        idx,
        rank=req.rank,
        category=req.category,
        state=req.state,
        course=req.course,
        year=req.year,
        quota=req.quota,
    )
    return {
        "query": {
            "rank": pred.rank,
            "category": pred.category,
            "state": pred.state,
            "course": pred.course,
            "year": pred.year,
        },
        "warnings": pred.warnings,
        "options": [
            {
                "kind": o.kind,
                "college": o.college,
                "course": o.course,
                "quota": o.quota,
                "state": o.state,
                "category": o.category,
                "year": o.year,
                "opening_rank": o.opening_rank,
                "closing_rank": o.closing_rank,
                "seats": o.seats,
                "source_id": o.source_id,
            }
            for o in pred.options
        ],
    }


@app.post("/ingest")
def ingest(force: bool = False) -> dict:
    global _index
    _index = build_index(force=force)
    return {"index_size": _index.size}


@app.post("/eval")
def run_eval() -> dict:
    from .eval_runner import run_evaluation

    return run_evaluation(get_index())