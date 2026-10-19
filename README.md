# RAG: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

**GitHub repository:** https://github.com/Sankalp0109/RAG_ASSIGNMENT_LMA

Two retrieval pipelines are compared over the same chunked corpus, the *2024 FIA Formula One Sporting Regulations*
(Issue 6, 106 pages, `corpus_document.pdf`):

| | Pipeline 1: Baseline | Pipeline 2: Re-Ranking |
|---|---|---|
| Stage 1 | `all-MiniLM-L6-v2` bi-encoder + FAISS cosine Top-10 | same Top-10 |
| Stage 2 | — | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores the 10 (query, chunk) pairs, re-sort |
| Context | Top-3 by cosine | Top-3 by cross-encoder score |
| Generator | Gemini `gemini-3.6-flash`, temperature 0, identical prompt | same |

## Files

| File | What it is |
|---|---|
| `RAG_Baseline_vs_Reranking.ipynb` | **Main deliverable.** Executed notebook with all code, outputs, rank tables, answers and latency numbers |
| `report/report.pdf` (+ `report/report.tex`, `report/acl_onecol.sty`) | 4-page single-column ACL-style report covering the four assigned areas: architecture, rank-shift & answer analysis, latency, scaling |
| `corpus_document.pdf` | The corpus |
| `src/chunking.py` | Article-aware chunker (also inlined in the notebook) |
| `src/queries.json` | The 52-question failure-mode test suite (category, gold sentence, ground truth, observations) |
| `src/build_notebook.py` | Generates the notebook from the chunker + test suite |
| `src/screen_queries.py`, `src/candidates.json` | Script used to screen candidate queries for rank shifts |
| `results/` | Saved outputs: rank table, latency, scaling, summary JSON, and `llm_cache.json` (cached Gemini answers) |
| `requirements.txt` | Python dependencies |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'GOOGLE_API_KEY=<your-gemini-key>' > .env     # .env is git-ignored
jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_Reranking.ipynb
```

The embedding and re-ranking models download from Hugging Face on first run. Any device works (CPU / Apple MPS / CUDA).

The Gemini free tier allows only ~20 requests/day per model, so LLM answers are cached in `results/llm_cache.json`
(with the latency measured when they were generated). Re-running the notebook re-uses them; delete that file to
regenerate fresh answers.

To rebuild the report: `cd report && pdflatex report && pdflatex report`.
