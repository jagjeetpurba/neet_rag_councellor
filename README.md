# NEET RAG Counsellor

A **private, locally hosted retrieval-augmented generation (RAG) assistant** for
NEET UG counselling. It answers questions over cut-off and seat-matrix data
**with citations**, and **refuses** ("not in the sources") when the retrieved
material does not support an answer, rather than guessing.

This is the core task (section 2.1) plus **Track A** (section 2.2-A, the college
predictor) from the CGPRR AI Developer take-home assignment.

Everything runs on open-weight models served locally. No data leaves the
machine, and no paid cloud LLM API is required.

---

## 1. What it does

| Capability | Where |
|---|---|
| Ingest cut-off tables, build a local index | `src/neet_rag/ingest.py` |
| Answer questions with source citations | `src/neet_rag/generate.py` |
| Refuse when the answer isn't in the sources | `src/neet_rag/retrieve.py` (refusal gate) + `generate.py` |
| NEET college predictor (rank + category + state) | `src/neet_rag/predictor.py` |
| Evaluation set with honest scoring | `eval/questions.json` + `src/neet_rag/eval_runner.py` |
| REST API | `src/neet_rag/app.py` |
| CLI | `src/neet_rag/cli.py` |

---

## 2. Requirements

- **Linux or macOS** (developed on an Apple M2, 16 GB — see §5 for the GPU target).
- **Python 3.10+** (tested on 3.13).
- **[Ollama](https://ollama.com)** running locally (serves the model and embeddings).

---

## 3. Quick start

```bash
# 1. Install Ollama (macOS)
brew install ollama                      # or: https://ollama.com/download

# 2. Pull the open-weight models (generation + embeddings)
ollama pull qwen2.5:7b                   # Apache 2.0
ollama pull nomic-embed-text             # Apache 2.0

# 3. Clone and set up the environment
git clone <this-repo> && cd neet-rag-counsellor
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                        # installs the package + deps

# 4. (Optional) regenerate synthetic seed data
python scripts/generate_synthetic.py

# 5. Build the index
neet-rag ingest

# 6. Ask a question that *is* in the sources -> cited answer
neet-rag ask "What was the closing rank for MBBS at AIIMS New Delhi in the Open category in 2024?"

# 7. Ask a question that is *not* -> refusal
neet-rag ask "What is the cut-off for MBBS at AIIMS Jodhpur in 2024?"

# 8. Run the evaluation and read the score table
neet-rag eval
```

### Run the predictor (Track A)

```bash
neet-rag predict --rank 2000 --category Open --state "West Bengal" --course MBBS --year 2024
# or via API
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"rank":2000,"category":"Open","state":"West Bengal","course":"MBBS"}'
```

### Run the API server

```bash
neet-rag serve      # http://127.0.0.1:8000  (docs at /docs)
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness + model/index summary |
| `/ask` | POST | `{"question": "..."}` -> cited answer or refusal |
| `/predict` | POST | rank/category/state/course/year -> options with sources |
| `/ingest` | POST | rebuild the index (`?force=true` to force) |
| `/eval` | POST | run the eval set, return the score table |

---

## 4. Architecture

```
           ┌────────────────────────── FastAPI (app.py) ─────────────────────────┐
           │   /ask                   /predict                 /ingest  /eval    │
           └──────┬───────────────────────┬─────────────────────────┬───────────┘
                  │                       │                         │
        ┌─────────▼─────────┐   ┌─────────▼─────────┐     ┌─────────▼─────────┐
        │  retrieve.py      │   │  predictor.py     │     │  ingest.py        │
        │  query -> embed   │   │  exact rank/cat   │     │  table -> passages│
        │  top-k cosine     │   │  filter over table│     │  -> embed -> index│
        │  refusal gate     │   │  (no LLM guess)   │     └─────────┬─────────┘
        └─────────┬─────────┘   └─────────┬─────────┘               │
                  │                       │                ┌────────▼─────────┐
        ┌─────────▼─────────┐             │                │  index.py         │
        │  generate.py       │◄────────────┘                │  numpy cosine     │
        │  LLM + citations   │   explanation w/ sources     │  passages.json    │
        │  refusal token     │                              │  embeddings.npz   │
        └─────────┬─────────┘                              └───────────────────┘
                  │
        ┌─────────▼─────────┐
        │  Ollama (local)    │   qwen2.5:7b        +  nomic-embed-text
        └───────────────────┘   (generation)          (embeddings)
```

Key design points:

- **Retrieval** is a plain **numpy cosine-similarity** index over embeddings —
  deliberately small and auditable. Swap for FAISS/Qdrant behind the same
  interface at scale; nothing above it changes.
- **Refusal is layered** so a wrong cut-off never reaches the user: a
  *coverage guard* (named-but-absent year/entity, or an out-of-scope attribute
  like fee/hostel) refuses before generation; a *similarity gate* refuses when
  nothing relevant is retrieved; a *refusal token* (`NOT_IN_SOURCES`) in the
  prompt; and a *citation audit* strips hallucinated `[source]` ids and refuses an
  answer with no valid citation.
- **Track A predictor is deterministic** (exact `rank <= closing_rank` and
  `category == X` filters over the table), never a model guessing a number. The
  LLM is used only to *explain* the deterministic list with citations.

---

## 5. Model & hardware assumptions

| | Development (this demo) | Deployment target (assumed) |
|---|---|---|
| **Model** | `qwen2.5:7b` (4-bit, Apache 2.0) | `Qwen2.5-7B-Instruct` (BF16, Apache 2.0) |
| **Serving** | Ollama (CPU + Metal) | vLLM |
| **Embeddings** | `nomic-embed-text` (Apache 2.0) | same |
| **Hardware** | Apple M2, 16 GB unified memory | **single NVIDIA RTX 6000-class workstation GPU, 48 GB VRAM** (assumed) |
| **Memory** | ~6 GB (4-bit weights + context) | ~18 GB VRAM (7B BF16 + KV cache) — about a third of the card, leaving headroom for batched serving |

The **same weights** run in both places; only the runtime and quantisation
differ, so the demo is representative of the deployed behaviour.

---

## 6. Evaluation

`eval/questions.json` holds **17 cases** (7 answerable + 5 refusal + 3 unsafe +
2 predictor), each marked in advance with its expected behaviour:

- **`answerable`** (7): must return a grounded answer that cites a real source
  and contains the correct figure.
- **`refusal`** (5): must refuse (data not present, out of domain, wrong year).
- **`unsafe`** (3): prompt injection / planted figures / guarantee requests —
  must **not** state the injected content (mirrors the assignment's §2.4 cases).
- **`predict`** (2): deterministic predictor must list the right options, and
  return *nothing* when the state/category has no data.

Run `neet-rag eval` to reproduce the score table. Results are printed honestly
(no masking of failures).

---

## 7. Data & privacy

- **Synthetic seed data** (`data/synthetic/neet_cutoffs.csv`) is generated
  deterministically by `scripts/generate_synthetic.py` and released as **CC0**.
  No real student, college, or government figures are used. *Real public data*
  being cleaned from MCC/state counselling PDFs can be dropped into
  `data/real/` (same CSV schema) and built with `NEET_DATA_DIR=data/real neet-rag ingest` — see `DESIGN_NOTE.md`.
- **Nothing leaves the machine**: the index (`index/`), model cache, and logs are
  all local; the only network calls are the one-time model pulls from Ollama's
  public registry.
- **Per-business separation** (for an office install): each business gets its own
  index directory, model served under its own Ollama instance/namespace, and its
  own logs — so NEET data is never mixed with Hirrx or editorial data. See the
  design note for detail.

---

## 8. AI tooling disclosure

AI coding tools were used to draft code and documentation (list the specific
tools you used here — e.g. a coding agent / LLM). All code has been reviewed
and can be walked through line by line.

## 9. Directories

```
data/synthetic/        synthetic cut-off CSV (CC0)
eval/questions.json    evaluation set
scripts/               data generation
src/neet_rag/          package (config, ingest, index, retrieve, generate, predictor, app, cli, eval_runner)
index/                 built index (embeddings.npz + passages.json) — gitignored
DESIGN_NOTE.md         two-page design note
```