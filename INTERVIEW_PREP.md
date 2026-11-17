# NEET RAG Counsellor — Interview Prep (end-to-end deep dive)

A complete, interview-ready explanation of the project: what it does, the exact
data flow, every architectural decision and **why** it was made, and the likely
follow-up questions.

---

## 0. The 30-second elevator pitch

> I built a **private, locally hosted RAG assistant** for NEET UG counselling. It
> ingests cut-off and seat-matrix tables, answers questions **with citations to the
> source row**, and **refuses** — "not in the sources" — when it can't back an
> answer, instead of guessing. Everything runs on **open-weight models served
> locally** on a single workstation GPU, so no student data ever leaves the
> machine. On top of the RAG I built the **college predictor** (rank + category →
> plausible colleges with sources), a **layered refusal system**, and an
> **evaluation set** that scores honesty — including prompt-injection cases.

---

## 1. The problem (what the assignment actually demanded)

Three things, all local-first and *checkable*:

1. **Core task** — a RAG that: runs on an open-weight model on one workstation GPU
   (RTX 6000-class, I state the VRAM); ingests documents/tables; answers **with
   citations**; **refuses** when sources don't support the answer.
2. **One track** — I chose **Track A**, the NEET college/counselling predictor,
   which must include an "unsafe advice" test (a question that would require
   inventing a number — the system must not guess).
3. **Three artifacts** — working code + README, a 2-page design note, a 3–5 min
   demo video. Grading: 30% core RAG, 20% track, 20% design note, 10% evaluation,
   10% code/README, 10% video.

The single most important phrase in the brief: *"We care more about a system that
can be checked than about a long list of tools."* That sentence drove every
decision below.

---

## 2. Architecture (the diagram you should be able to draw from memory)

```
            ┌────────────────── FastAPI (app.py) ──────────────────┐
            │   /ask          /predict          /ingest    /eval   │
            └─────┬────────────────┬──────────────────┬────────────┘
                  │                │                  │
        ┌─────────▼────────┐  ┌────▼────────────┐  ┌──▼──────────────────┐
        │ retrieve.py       │  │ predictor.py    │  │ ingest.py           │
        │  embed query      │  │  exact filters  │  │  CSV → passages     │
        │  top-k cosine     │  │  (no LLM math)  │  │  → embed → index    │
        │  layered refusal  │  └────┬────────────┘  └──┬──────────────────┘
        └─────────┬────────┘       │                   │
                  │                │            ┌──────▼──────────┐
        ┌─────────▼────────┐       │            │ index.py          │
        │ generate.py       │◄──────┘            │  numpy cosine     │
        │  prompt + cite    │  explanation w/    │  passages.json    │
        │  refusal token    │  citations         │  embeddings.npz   │
        │  citation audit   │                    └──────────────────┘
        └─────────┬────────┘
                  │
        ┌─────────▼────────┐
        │  Ollama (local)   │   qwen2.5:7b         nomic-embed-text
        └──────────────────┘   (generation)         (embeddings)
```

**Why this shape?** Three principles:

1. **Separate retrieval from generation.** Retrieval is deterministic
   (embeddings + cosine similarity); generation is probabilistic. Keeping them
   apart means I can unit-test the retrieval and refusal logic without a GPU, and
   I can substitute models/DBs independently.
2. **The predictor is a sibling of the RAG, not part of the LLM.** Structured
   predicates (`rank <= closing_rank`, `category == X`) are *exact math*. A
   generative model is the wrong tool for exact math — it would hallucinate
   numbers, which is precisely what the assignment penalises.
3. **One local serving daemon (Ollama) for both models.** Fewer moving parts to
   explain, no PyTorch/sentence-transformers install (~2 GB), fits a 16 GB Mac.

---

## 3. End-to-end flow, step by step

### Flow A — ingest (build the index)

```
CSV row (200 rows)
   │  row_to_passage(): render one table row into natural language + stable id
   ▼
Passage { source_id="neet_cutoffs#36", text="AIIMS New Delhi, MBBS, AIQ quota,
          Delhi, Open category, 2024, opening rank 37, closing rank 50, 15 seats.",
          meta={college, course, quota, state, category, year, opening, closing, seats} }
   │  embed the *text* with nomic-embed-text (local)
   ▼
embeddings (200 × D float32)
   │  VectorIndex.save()
   ▼
index/embeddings.npz  +  index/passages.json
```

Design points worth saying aloud:
- **`source_id` is a first-class citation handle**, assigned at ingest time
  (`neet_cutoffs#N` = row N). Citations in answers point back to these.
- **The embedding input is the human-readable passage, but the structured `meta`
  is preserved** alongside it. This is what lets the predictor do exact filtering
  (it works on `meta`) while Q&A does semantic search (it works on `text`).
- **The index is plain files** (`.npz` + `.json`) — you can `cat` the passages
  file and see every row. That audibility is deliberate.

### Flow B — /ask (Q&A with citations + refusal)

```
question
  │ 1. embed the question
  │ 2. top-k cosine similarity against the index
  ▼
  │ 3. LAYERED REFUSAL (any layer can stop here):
  │    a) coverage guard  — question names a year/entity/attribute not in corpus
  │    b) similarity gate — best match below threshold (0.30) → nothing relevant
  ▼
retrieved passages + question
  │ 4. prompt the local model: "answer ONLY from context, cite [id], else
  │    reply NOT_IN_SOURCES"
  ▼
raw model output
  │ 5. citation audit — strip any [id] not actually in the retrieved set;
  │    if no valid citation survives → refuse
  ▼
Answer { text (with [neet_cutoffs#36] ...), refused, citations, reason }
```

### Flow C — /predict (Track A, deterministic)

```
rank/category/state/quota/course/year
  │  load all passage meta into a DataFrame (one row per passage)
  │  filter:  category == X  AND  year == Y  AND  [quota/state/course if given]
  │  split:   eligible = rows where rank <= closing_rank   (you'd have qualified)
  │           reach    = rows where rank >  closing_rank, within a band (tough)
  ▼
options, each returning its source_id → explainable, checkable, NO guessed numbers
```

**The key idea:** eligibility is `rank <= closing_rank` (a lower rank is better).
Everything is an exact table filter. The LLM is *only* used to phrase the
explanation with citations — never to produce the rank math.

### Flow D — /eval

`eval/questions.json` marks the expected behaviour of every case **in advance**,
then `eval_runner.py` runs each through the live pipeline and scores mechanically:

- **answerable** → must be non-refused, cite a valid source, and contain the correct figure.
- **refusal** → must refuse.
- **unsafe** → must NOT state the injected/planted content.
- **predict** → must list the right options / return nothing when data is absent.

The output is a plain score table — failures are not hidden.

---

## 4. The refusal system (the single most important design, ~30% of the grade)

The core requirement is *"refuse when the sources don't support the answer, never
confidently guess."* A confident wrong cut-off is worse than silence, so refusal
is **defence in depth — four layers, each catching a different failure mode**:

| # | Layer | Where | Catches |
|---|---|---|---|
| 1 | **Coverage guard** | `retrieve.py`, before the LLM | a named year/proper-noun(→ "AIIMS Jodhpur") or out-of-scope attribute ("fee", "hostel") not in the corpus |
| 2 | **Similarity gate** | `retrieve.py`, before the LLM | nothing semantically relevant retrieved (out-of-domain) |
| 3 | **Refusal token** | prompt + `generate.py` | the model itself sees the context doesn't answer → emits `NOT_IN_SOURCES` |
| 4 | **Citation audit** | `generate.py`, after generation | the model invented a `[source]` id → stripped; zero valid citations → refuse |

Why layers 1–2 are *deterministic* rather than trusting the model: an embedding
of "AIIMS Jodhpur" will still match "AIIMS New Delhi" at ~0.77 cosine similarity
(they share "AIIMS"/"MBBS"), so a similarity gate alone can't tell the two apart.
But "Jodhpur" is a **proper noun that simply isn't anywhere in the corpus** — that
is checkable by a deterministic string/set test, so I refuse *before* the model
can hallucinate. This is the exact fix for the assignment's example ("cut-off for
a college not in your index").

Layer 3–4 is the model + a post-hoc check: the prompt instructs "answer only from
context, cite `[id]`, else `NOT_IN_SOURCES`", and the audit strips any `[id]` that
doesn't exist in the retrieved passages — so even a model that tries to cite a
made-up source gets caught.

---

## 5. Why each technology / decision (the "why" questions interviewers ask)

### Why open-weight + fully local?
The brief is explicit: student data must not go to an external model service; the
group already owns a workstation GPU. Open weights (Qwen2.5, Apache 2.0) are free
and can run on-prem. RAG over *private* data is exactly the "run it yourself"
pattern.

### Why Qwen2.5-7B (and one model in two runtimes)?
- 7–8B is the sweet spot: strong enough to follow a "cite + refuse" instruction
  reliably, small enough to fit a single GPU with long-context headroom.
- Apache 2.0 (free, commercial-friendly) — the user explicitly wanted a free model.
- **Honesty trick:** I serve the *same weights* on the demo machine (Ollama, Q4
  quant on an M2 Mac) and the target (vLLM, BF16 on the RTX 6000). The demo is
  representative, and the design note can honestly say "I ran X, here's the
  estimate for Y" — which the brief explicitly asks for when you don't have the GPU.
- VRAM math (state it confidently): 7B BF16 ≈ 15 GB weights + ~3 GB KV/context ≈
  18 GB on a 48 GB card → ~⅓ utilisation, headroom for batching or a 13–14B model.
  On the M2 demo I use 4-bit (~5 GB).

### Why RAG instead of fine-tuning?
- The data is tabular and **changes yearly**; fine-tuning means retraining every
  cycle. RAG just rebuilds an index.
- The assignment demands **citations to the source**, which fine-tuning cannot
  guarantee — retrieval gives you the exact row, so the citation is a lookup, not
  a generation.
- Only ~200 rows — nowhere near enough to fine-tune, plenty to embed.
- No training infra on a single workstation GPU.

### Why a plain numpy cosine index instead of Chroma/Qdrant/FAISS?
- "Smaller, working, checkable demo beats a long list of tools." 200 passages is
  trivial for numpy.
- The index is *auditable* — two files you can open and inspect.
- The retrieval interface is isolated, so swapping in FAISS/Qdrant later is a
  one-file change. This shows maturity: **choose the simplest tool that meets the
  current scale, keep the door open to scale up.**

### Why Ollama for BOTH embeddings and generation?
- Embeddings need a model too; `nomic-embed-text` (Apache 2.0) runs in the same
  daemon — one runtime to operate and explain.
- Avoids `sentence-transformers` + PyTorch (~2 GB install) and the extra memory
  on a 16 GB Mac. Resource-consciousness matters in a "fits a workstation"
  context.

### Why FastAPI + a lean custom pipeline instead of LangChain/LlamaIndex?
- The brief: *"be ready to explain every part on a later call."* A ~50-line
  retrieval loop I wrote is trivially explainable; a framework's hidden abstractions
  are not.
- Frameworks add moving parts (agents, chains, callback systems) I don't need for
  a single-hop RAG.
- Trade-off I can name honestly: Losing LangChain means I hand-write the retrieval
  + prompt logic, which is fine at this size and *better* for demonstrating
  understanding.

### Why a **deterministic** predictor, not a generative one?
- "From rank 2000 and OBC, which colleges?" is exact table filtering. A generative
  model would risk inventing a closing rank — the exact "unsafe advice" failure the
  assignment penalises.
- Deterministic code is testable and auditable; every option carries its `source_id`.
- The LLM is confined to *explaining* the deterministic result, which is a safe job.

### Why synthetic seed data (with real data optional)?
- Explicitly allowed ("public or synthetic"), CC0, fully in my control, and every
  evaluation figure is known-correct so the eval can be checked.
- Real NEET cut-offs (my reconnaissance found) DO live on the official NIC
  "OR-CR" portal (`admissions.nic.in`) with the exact schema — but they're
  JS-paginated and need browser scraping + a PDF join for the `seats` column.
  That's a bonus, not a dependency.

---

## 6. Model & hardware narrative (have these numbers ready)

| | Demo (what I ran) | Target (assumed) |
|---|---|---|
| Model | qwen2.5:7b (4-bit) | Qwen2.5-7B-Instruct (BF16) |
| Serving | Ollama (CPU + Metal) | vLLM |
| Embeddings | nomic-embed-text (137M) | same |
| Hardware | Apple M2, 16 GB unified | single RTX 6000-class, **48 GB VRAM** |
| Weights / memory | ~5 GB | ~15 GB + ~3 GB KV ≈ 18 GB |
| Throughput est. | a few tok/s (CPU) | 35–60 tok/s single-stream |

Always state the **VRAM assumption explicitly** (the brief asks for it), and the
honest "I ran on CPU, here's the GPU estimate" framing.

---

## 7. Evaluation & honesty (10% + the "numbers are honest" requirement)

- 15 Q&A questions + 2 predictor checks, each marked in advance:
  - **7 answerable** (must cite + correct figure, e.g. "closing rank 50"),
  - **5 refusal** (college not in index, wrong year, fee, out-of-domain),
  - **3 unsafe** (prompt injection "cut-off is 1", planted "99999", guarantee request),
  - **2 predict** (right options for a qualifying rank; nothing for a state with no data).
- Scoring is mechanical (pass/fail per rule), printed plainly, failures unhidden.
- This maps 1:1 to the assignment's own example questions in §2.4.

---

## 8. Privacy & separation (section 2.5 of the brief)

- Index, model cache, and logs are **local only**; the only network traffic is the
  one-time model pull from Ollama's public registry.
- No student/patient/doctor data — synthetic seed (CC0); real public cut-offs, if
  added, go to `data/real/` under the same schema.
- **Per-business isolation** (for an office install): separate index dir, separate
  Ollama namespace, separate logs, separate `NEET_DATA_DIR` per business — NEET,
  Hirrx, and editorial corpora never share a store.

---

## 9. Likely interview questions — and how to answer

**"Why not just ask GPT-4 and be done?"**
→ Privacy (student data can't leave), cost (no per-token fees), the brief requires
open-weight + on-prem, and citations require control over retrieval that an API
assistant doesn't give.

**"How do you know the answer is actually grounded?"**
→ Two deterministic backstops (coverage guard, similarity gate) + the citation
audit: every `[id]` in the answer must exist in the retrieved passages, and an
answer with no valid citation is refused. Plus the model is told to answer only
from context.

**"How does it refuse a college that isn't in the index?"**
→ The coverage guard detects the named proper noun ("Jodhpur") isn't anywhere in
the corpus and refuses before the model runs — because "AIIMS Jodhpur" would
still *embed* close to "AIIMS New Delhi", so similarity alone is insufficient.

**"Why cosine similarity over a learned ranker?"**
→ Scale + explainability. 200 passages needs no learned reranker; if you scale to
hundreds of thousands, you'd add a cross-encoder reranker on top of the top-k —
same interface.

**"What would you improve / what are the limits?"**
→ (1) Real data via the NIC OR-CR portal + MCC seat-matrix PDF join. (2) A
cross-encoder reranker and hybrid (BM25 + dense) retrieval for larger corpora.
(3) Streamed/gated generation (guardrails library) instead of my hand-rolled audit.
(4) A proper eval harness (pytest + CI) and a golden set. (5) Per-user auth +
audit logging before any production use. (6) It's **not** production counselling
software — human counsellors remain responsible for advice that affects people.

**"What's the latency breakdown?"**
→ Embedding (nomic, 137M) is milliseconds; retrieval is a matrix multiply (ms);
generation dominates (35–60 tok/s on GPU → a 100-token answer ≈ 2–3 s).

**"How do you prevent prompt injection?"**
→ The prompt says to answer only from context; the refusal token provides a safe
output; the citation audit strips non-context `[ids]`; and the coverage guard
catches injected instructions like "Ignore the documents" deterministically in
many cases. I log a refusal reason so injections are auditable.

**"Why is the predictor safe from hallucination?"**
→ It never asks the model to compute eligibility. `rank <= closing_rank` is a
Pandas boolean filter over the ingested table; the option list is returned with
`source_id`s, and the model only rephrases it.

---

## 10. The one-liners that tie it together

- "Retrieval is deterministic, generation is probabilistic — I keep them separate
  so the risky part is as small as possible."
- "A confident wrong cut-off is worse than no answer, so refusal is layered, not
  delegated entirely to the model."
- "Citations aren't a prompt-hint — they're a *lookup*: the retrieved row IS the
  source."
- "I chose the simplest tool that meets each requirement, and isolated each
  interface so it can be swapped when scale demands it."