# Demo video script (3–5 minutes)

One success case (cited answer) + one failure/refusal case are required;
everything else is supporting. Record the terminal (or the web UI) with a
screen recorder, then upload unlisted to YouTube or a cloud drive.

## Order to record

1. **Setup (20 s).** `ollama serve` is running; show `ollama list`
   (both models present), then `neet-rag ingest` already done -> show
   `index/` has `embeddings.npz` + `passages.json`.

2. **Good answer with citations (60–90 s).**
   ```bash
   neet-rag ask "What was the closing rank for MBBS at AIIMS New Delhi in the Open category in 2024?"
   ```
   Narration: point at the cited `[neet_cutoffs#36]` and the sourced figure
   (closing rank 50), and note it came only from the indexed table.

3. **Predictor / Track A (60 s).**
   ```bash
   neet-rag predict --rank 2000 --category Open --state "West Bengal" --course MBBS --year 2024 --quota State
   ```
   Narration: rank/category/state -> deterministic options with source ids,
   "eligible vs reach", no guessed numbers.

4. **Refusal case (45–60 s).** Pick one and narrate *why* it refuses:
   ```bash
   neet-rag ask "What is the cut-off for MBBS at AIIMS Jodhpur in 2024?"     # college not in index
   neet-rag ask "What is the annual tuition fee for MBBS at Medical College Kolkata?"  # attribute not in sources
   ```
   Narration: "Jodhpur/`fee` isn't in the corpus, so it says *not in the
   sources* instead of inventing a number."

5. **Evaluation (30–45 s).**
   ```bash
   neet-rag eval
   ```
   Narration: read the score table aloud, note the passed refusal/unsafe cases.

Keep it tight; 3–5 minutes total. If a case fails on camera, say so — the
assignment asks for honest limits.