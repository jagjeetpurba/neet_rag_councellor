"""Command-line interface.

Usage (from repo root, with the venv active):
  python -m neet_rag.cli ingest                # build the index
  python -m neet_rag.cli ask "<question>"      # cited answer / refusal
  python -m neet_rag.cli predict --rank 2000 --category Open --state "West Bengal"
  python -m neet_rag.cli eval                  # run the evaluation set
  python -m neet_rag.cli serve                 # start the FastAPI server
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from . import config
from .generate import ensure_generator_model
from .index import VectorIndex
from .ingest import build_index

console = Console()


def _load_index() -> VectorIndex:
    if not (config.INDEX_DIR / "embeddings.npz").exists():
        console.print("[yellow]index not found; building it first...[/]")
        idx = build_index()
    else:
        idx = VectorIndex.load(config.INDEX_DIR)
    return idx


def cmd_ingest(force: bool) -> None:
    build_index(force=force)


def cmd_ask(question: str) -> None:
    from .generate import answer

    ensure_generator_model()
    idx = _load_index()
    a = answer(question, idx)

    console.print(f"\n[bold]Q:[/] {question}\n")
    if a.refused:
        console.print(f"[red]{a.text}[/]")
        console.print(f"[dim]reason: {a.reason}[/]")
    else:
        console.print(a.text)
        console.print(f"\n[green]citations: {', '.join(a.citations)}[/]")


def cmd_predict(args: argparse.Namespace) -> None:
    from .predictor import predict

    idx = _load_index()
    pred = predict(
        idx,
        rank=args.rank,
        category=args.category,
        state=args.state,
        course=args.course,
        year=args.year,
        quota=args.quota,
    )

    console.print(
        f"\n[bold]Rank {pred.rank}, {pred.category}"
        + (f", {pred.state}" if pred.state else "")
        + (f", {pred.course}" if pred.course else "")
        + (f", {pred.year}" if pred.year else "")
        + "[/]\n"
    )
    for w in pred.warnings:
        console.print(f"[yellow]warning: {w}[/]")

    if not pred.options:
        console.print("[red]no options found in the sources.[/]")
        return

    table = Table(title="Plausible options", show_lines=False)
    for col in ("kind", "college", "course", "quota", "category", "year", "opening", "closing", "seats", "source"):
        table.add_column(col)
    for o in pred.options:
        table.add_row(
            o.kind, o.college, o.course, o.quota, o.category,
            str(o.year), str(o.opening_rank), str(o.closing_rank),
            str(o.seats), o.source_id,
        )
    console.print(table)


def cmd_eval() -> None:
    from .eval_runner import _fmt, run_evaluation

    ensure_generator_model()
    _load_index()
    res = run_evaluation()
    console.print(_fmt(res))


def cmd_serve() -> None:
    import uvicorn

    ensure_generator_model()
    _load_index()
    uvicorn.run("neet_rag.app:app", host="127.0.0.1", port=8000, reload=False)


def main() -> None:
    p = argparse.ArgumentParser(prog="neet-rag", description="NEET RAG Counsellor")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest").add_argument("--force", action="store_true")

    a = sub.add_parser("ask")
    a.add_argument("question", nargs="+")

    pr = sub.add_parser("predict")
    pr.add_argument("--rank", type=int, required=True)
    pr.add_argument("--category", required=True)
    pr.add_argument("--state", default=None)
    pr.add_argument("--course", default=None)
    pr.add_argument("--year", type=int, default=None)
    pr.add_argument("--quota", default=None, help="AIQ or State")

    sub.add_parser("eval")
    sub.add_parser("serve")

    args = p.parse_args()

    if args.cmd == "ingest":
        cmd_ingest(args.force)
    elif args.cmd == "ask":
        cmd_ask(" ".join(args.question))
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "eval":
        cmd_eval()
    elif args.cmd == "serve":
        cmd_serve()


if __name__ == "__main__":
    main()