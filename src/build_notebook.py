"""Builds RAG_Baseline_vs_Reranking.ipynb from the cell sources below (run once, then execute the notebook)."""
import json
import pprint
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
chunking_src = (ROOT / "src" / "chunking.py").read_text()
queries = json.loads((ROOT / "src" / "queries.json").read_text())

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# RAG Assignment: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

**GitHub repository:** https://github.com/Sankalp0109/RAG_ASSIGNMENT_LMA

**Corpus:** *2024 FIA Formula One Sporting Regulations* (Issue 6, published 30 April 2024, 106 pages): `corpus_document.pdf`.
It is a dense regulatory document full of general rules followed by "unless…", "except…", "may not…" clauses,
cross-references between articles, and repeated vocabulary (*safety car, pit lane, penalty, tyres*) across many
unrelated articles, which makes it a good stress test for dense retrieval.

| Component | Choice |
|---|---|
| Chunking | Article-aware: one chunk per sub-article (e.g. `55.8`), long ones split at ≤900 chars with 150-char overlap |
| Bi-Encoder (dense retrieval) | `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` on L2-normalised vectors (= cosine similarity) |
| Cross-Encoder (re-ranker) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini (`gemini-3.6-flash`, temperature 0), same prompt for both pipelines |

**Pipelines**
* **Pipeline 1 (Baseline):** query → bi-encoder → FAISS Top-10 → take Top-3 → LLM → *Answer 1*
* **Pipeline 2 (Re-Ranking):** the same Top-10 → cross-encoder scores all 10 (query, chunk) pairs → re-sort → Top-3 → LLM (identical prompt) → *Answer 2*

**Notebook outline**
1. Setup & configuration
2. Load and chunk the PDF
3. Bi-encoder indexing (vector store)
4. Retrieval pipelines (baseline + re-ranking)
5. LLM answer generation
6. Failure-mode test suite (rank analysis on all queries)
7. Deep dive: rank shifts + Answer 1 vs Answer 2 for each failure query
8. Latency profiling
9. Scaling experiment: why not run the cross-encoder over the whole corpus?
10. Summary
""")

md("## 1. Setup & configuration")
code(r"""
# Install dependencies (uncomment on a fresh environment)
# %pip install torch sentence-transformers faiss-cpu pymupdf numpy pandas google-genai

import os, re, time, json, textwrap, statistics
from pathlib import Path
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

# ---- configuration -------------------------------------------------------------
PDF_PATH        = "corpus_document.pdf"
BI_ENCODER      = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL       = "gemini-3.6-flash"
TOP_K_RETRIEVE  = 10   # candidates fetched from the vector store
TOP_K_CONTEXT   = 3    # chunks passed to the LLM
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ---- API key: read from a local .env file (kept out of git) or the environment ----
def load_env(path=".env"):
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
load_env()
assert os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"), "Set GOOGLE_API_KEY in .env"

torch.manual_seed(0); np.random.seed(0)
print("torch", torch.__version__, "| device:", DEVICE)
""")

md(r"""
## 2. Load and chunk the PDF

The regulations are structured as *Article → sub-article → lettered clauses* (e.g. **55.8 a)–h)**).
Fixed-size chunking would cut rules in the middle of an exception list, so the chunker:

1. extracts text with PyMuPDF and strips the repeated page header/footer;
2. detects article headings (`36)` + `REFUELLING`) and sub-article numbers (`36.1`), ignoring cross-references such as
   "see Article 54.3" by requiring the number to belong to the current article and to increase monotonically;
3. creates one unit per sub-article, merges tiny neighbouring units of the same article, and splits long units
   (≤ 900 characters, about the 256-token window of MiniLM) on sentence/line boundaries with a 150-character overlap;
4. keeps metadata (article number/title, sub-article ids, pages) so we can check *which rule* each chunk holds.

Both pipelines use exactly the same chunks.
""")
code(chunking_src.replace('"""PDF loading + article-aware chunking for the F1 Sporting Regulations corpus."""\n', "")
     .replace("import re\nimport fitz  # PyMuPDF\n", "").strip())
code(r"""
chunks = chunk_document(PDF_PATH)
texts = [c["text"] for c in chunks]
lens = [len(t) for t in texts]
print(f"{len(chunks)} chunks | {len({s for c in chunks for s in c['subs']})} sub-articles | "
      f"chars per chunk: min={min(lens)}, median={int(np.median(lens))}, max={max(lens)}")

def label(c):
    # Short label for a chunk, e.g. "55.8 (Safety Car)"
    sub = c["subs"][0] if c["subs"] else c["article"]
    return f"{sub} ({c['title'][:28]})"

pd.DataFrame([{"chunk_id": c["chunk_id"], "label": label(c), "pages": c["pages"], "text": c["text"][:110] + "…"}
              for c in chunks[::60]])
""")

md(r"""
## 3. Bi-encoder indexing (vector store)

Every chunk is embedded **once, offline** by the bi-encoder. Vectors are L2-normalised so that the inner product in a
FAISS `IndexFlatIP` equals cosine similarity. (The corpus is small, so exact search is used; for millions of chunks an
ANN index such as HNSW/IVF would replace it.)
""")
code(r"""
bi_encoder = SentenceTransformer(BI_ENCODER, device=DEVICE)
cross_encoder = CrossEncoder(CROSS_ENCODER, device=DEVICE)

t0 = time.perf_counter()
chunk_emb = bi_encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False).astype("float32")
index = faiss.IndexFlatIP(chunk_emb.shape[1])
index.add(chunk_emb)
index_time = time.perf_counter() - t0
print(f"Indexed {index.ntotal} chunks, dim={chunk_emb.shape[1]}, in {index_time:.2f}s (one-off cost)")

# warm-up both models so the first timed call does not include lazy initialisation
_ = bi_encoder.encode(["warm up"], normalize_embeddings=True)
_ = cross_encoder.predict([("warm up", texts[0])])
""")

md(r"""
## 4. Retrieval pipelines

* `baseline_retrieve` = **Pipeline 1, steps a–b**: embed the query, take the Top-10 chunks by cosine similarity.
* `rerank` = **Pipeline 2, steps a–c**: score each of the *same* 10 (query, chunk) pairs jointly with the cross-encoder
  and re-sort by that score.

Each function also returns its run time so we can profile latency later.
""")
code(r"""
def sync():
    # MPS/CUDA kernels run asynchronously; synchronise so the timings are honest
    if DEVICE == "mps": torch.mps.synchronize()
    elif DEVICE == "cuda": torch.cuda.synchronize()

def baseline_retrieve(query, k=TOP_K_RETRIEVE):
    # Pipeline 1: bi-encoder + vector search
    t0 = time.perf_counter()
    q = bi_encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    sync(); t1 = time.perf_counter()
    scores, ids = index.search(q, k)
    t2 = time.perf_counter()
    cands = [{"chunk_id": int(i), "bi_rank": r + 1, "bi_score": float(s)}
             for r, (i, s) in enumerate(zip(ids[0], scores[0]))]
    return cands, {"embed_ms": (t1 - t0) * 1e3, "search_ms": (t2 - t1) * 1e3}

def rerank(query, cands):
    # Pipeline 2: cross-encoder over the same Top-10 candidates
    t0 = time.perf_counter()
    ce_scores = cross_encoder.predict([(query, texts[c["chunk_id"]]) for c in cands], batch_size=len(cands))
    sync(); t1 = time.perf_counter()
    out = [dict(c, ce_score=float(s)) for c, s in zip(cands, ce_scores)]
    out.sort(key=lambda c: c["ce_score"], reverse=True)
    for r, c in enumerate(out):
        c["ce_rank"] = r + 1
    return out, {"rerank_ms": (t1 - t0) * 1e3}

# quick sanity check
cands, t = baseline_retrieve("What is the pit lane speed limit?")
reranked, t2 = rerank("What is the pit lane speed limit?", cands)
print([label(chunks[c["chunk_id"]]) for c in cands[:3]], t)
print([label(chunks[c["chunk_id"]]) for c in reranked[:3]], t2)
""")

md(r"""
## 5. LLM answer generation

Both pipelines use the **identical prompt**; the only difference is which three chunks fill the context. The model is told
to answer strictly from the context and to cite article numbers, so a missing chunk shows up as an omitted clause, a
wrong rule, or an admission that the context is insufficient, rather than the model quietly using what it already
knows about F1.
""")
code(r"""
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

PROMPT_TEMPLATE = '''You are an assistant answering questions about the 2024 FIA Formula One Sporting Regulations.
Answer the question using ONLY the context passages below. Do not use outside knowledge.
Cite the article numbers you rely on. If the passages do not contain the information needed,
say "The provided context does not answer this." Keep the answer to at most 4 sentences.

Context:
{context}

Question: {question}
Answer:'''

def build_context(cands):
    return "\n\n".join(f"[Passage {i + 1} | Article {label(chunks[c['chunk_id']])}]\n{texts[c['chunk_id']]}"
                       for i, c in enumerate(cands[:TOP_K_CONTEXT]))

# The free Gemini tier allows only ~20 requests/day per model, so every response is cached on disk together with
# the latency measured when it was generated. Re-running the notebook re-uses these answers (and their timings)
# instead of spending quota; delete results/llm_cache.json to force fresh calls.
import hashlib
CACHE_PATH = Path("results/llm_cache.json")
llm_cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

def generate_answer(question, cands, retries=4):
    prompt = PROMPT_TEMPLATE.format(context=build_context(cands), question=question)
    key = hashlib.sha256(f"{LLM_MODEL}\n{prompt}".encode()).hexdigest()
    if key in llm_cache:
        return llm_cache[key]["text"], llm_cache[key]["latency_ms"]
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(
                model=LLM_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0))
            text, ms = resp.text.strip(), (time.perf_counter() - t0) * 1e3
            llm_cache[key] = {"text": text, "latency_ms": ms, "question": question}
            CACHE_PATH.parent.mkdir(exist_ok=True)
            CACHE_PATH.write_text(json.dumps(llm_cache, indent=1))
            return text, ms
        except Exception as e:  # rate limits / transient network errors
            wait = 5 * (attempt + 1)
            print(f"LLM error ({type(e).__name__}: {str(e)[:80]}), retrying in {wait}s")
            time.sleep(wait)
    return "[LLM call failed]", float("nan")

print(generate_answer("What is the pit lane speed limit?", reranked)[0])
""")

md(r"""
## 6. Failure-mode test suite

Each query has a **gold sub-article** that holds the correct answer (checked by hand against the PDF) and is tagged with
the failure mode it is designed to probe:

* **Keyword trap / distractor**: the query shares words with an irrelevant passage, while the answer is phrased differently.
* **Exception / corner case**: a general rule exists, but the answer depends on an "unless / except" clause.
* **Negation / constraint**: "not allowed", "prohibited", "may not" questions.
* **Buried chunk**: the vital chunk exists in the Top-10 but sits at rank 4–10.

For every query we record the gold chunk's rank under the bi-encoder (Pipeline 1) and after cross-encoder re-ranking
(Pipeline 2). Because one sub-article can span several chunks, the gold is pinned to the *exact sentence* that answers
the question (`gold_text`): a chunk is gold if it contains that sentence (whitespace-normalised).
""")
code("QUERIES = " + pprint.pformat(queries, width=118, sort_dicts=False))
code(r"""
def norm(t):
    return " ".join(t.split()).lower()

def is_gold(chunk_id, q):
    return any(norm(g) in norm(texts[chunk_id]) for g in q["gold_text"])

for q in QUERIES:  # every gold sentence must exist in the corpus
    assert any(is_gold(i, q) for i in range(len(chunks))), q["id"]

def best_rank(ranked, q, key):
    ranks = [c[key] for c in ranked if is_gold(c["chunk_id"], q)]
    return min(ranks) if ranks else None   # None = gold not in the Top-10 at all

results = {}
rows = []
for q in QUERIES:
    cands, t_b = baseline_retrieve(q["query"])
    reranked, t_r = rerank(q["query"], cands)
    results[q["id"]] = {"cands": cands, "reranked": reranked}
    b, r = best_rank(cands, q, "bi_rank"), best_rank(reranked, q, "ce_rank")
    rows.append({"id": q["id"], "category": q["category"], "query": q["query"], "gold": ", ".join(q["gold"]),
                 "baseline_rank": b, "reranked_rank": r,
                 "shift": f"#{b} → #{r}" if b else "not in Top-10",
                 "base_top3": bool(b and b <= 3), "rerank_top3": bool(r and r <= 3)})
rank_df = pd.DataFrame(rows)
rank_df["baseline_rank"] = rank_df["baseline_rank"].astype("Int64")
rank_df["reranked_rank"] = rank_df["reranked_rank"].astype("Int64")
rank_df
""")
code(r"""
def mrr(col, df=None):
    df = rank_df if df is None else df
    return np.mean([0 if pd.isna(v) else 1 / v for v in df[col]])

summary = pd.DataFrame({
    "metric": ["Hit@3 (gold in context)", "MRR@10", "Hit@1"],
    "Pipeline 1: Baseline": [rank_df.base_top3.mean(), mrr("baseline_rank"), (rank_df.baseline_rank == 1).fillna(False).mean()],
    "Pipeline 2: Re-ranked": [rank_df.rerank_top3.mean(), mrr("reranked_rank"), (rank_df.reranked_rank == 1).fillna(False).mean()],
}).round(3)
in_top10 = rank_df.baseline_rank.notna()
print(f"{len(rank_df)} queries | gold inside Top-10 for {int(in_top10.sum())} "
      f"(re-ranking can only reorder these) | rescued into Top-3 by re-ranking: "
      f"{int(((~rank_df.base_top3) & rank_df.rerank_top3).sum())} | "
      f"pushed out of Top-3 by re-ranking: {int((rank_df.base_top3 & ~rank_df.rerank_top3).sum())}")
summary
""")
code(r"""
# Breakdown per failure category
per_cat = rank_df.groupby("category").apply(lambda d: pd.Series({
    "queries": len(d),
    "Hit@3 baseline": d.base_top3.mean(), "Hit@3 re-ranked": d.rerank_top3.mean(),
    "MRR@10 baseline": mrr("baseline_rank", d), "MRR@10 re-ranked": mrr("reranked_rank", d),
}), include_groups=False).round(2)
per_cat
""")
code(r"""
# All queries whose gold chunk moved across the Top-3 boundary (the cases that change what the LLM sees)
moved = rank_df[rank_df.base_top3 != rank_df.rerank_top3][["id", "category", "query", "gold", "shift"]]
moved
""")

md(r"""
## 7. Deep dive: rank shifts and Answer 1 vs Answer 2

For each failure query we show the full Top-10 with both ranks (★ = gold chunk), then generate **Answer 1** (baseline Top-3)
and **Answer 2** (re-ranked Top-3) with the identical prompt.
""")
code(r"""
llm_log = []

def show_query(qid):
    q = next(x for x in QUERIES if x["id"] == qid)
    res = results[qid]
    print("=" * 110)
    print(f"[{q['id']}] {q['category']}\nQ: {q['query']}\nGround truth ({', '.join(q['gold'])}): {q['truth']}")
    print("=" * 110)
    ce_by_id = {c["chunk_id"]: c for c in res["reranked"]}
    table = pd.DataFrame([{
        "gold": "★" if is_gold(c["chunk_id"], q) else "",
        "chunk": label(chunks[c["chunk_id"]]),
        "bi_rank": c["bi_rank"], "bi_score": round(c["bi_score"], 3),
        "ce_rank": ce_by_id[c["chunk_id"]]["ce_rank"], "ce_score": round(ce_by_id[c["chunk_id"]]["ce_score"], 2),
        "text": texts[c["chunk_id"]][:90].replace("\n", " ") + "…"} for c in res["cands"]])
    display(table)
    a1, t1 = generate_answer(q["query"], res["cands"])
    a2, t2 = generate_answer(q["query"], res["reranked"])
    llm_log.extend([t1, t2])
    print("ANSWER 1 (baseline Top-3:", [label(chunks[c["chunk_id"]]) for c in res["cands"][:3]], ")")
    print(textwrap.fill(a1, 110), "\n")
    print("ANSWER 2 (re-ranked Top-3:", [label(chunks[c["chunk_id"]]) for c in res["reranked"][:3]], ")")
    print(textwrap.fill(a2, 110))
    q["answer_1"], q["answer_2"] = a1, a2
""")

for q in sorted([q for q in queries if q.get("deep_dive")], key=lambda q: bool(q.get("regression"))):
    if q.get("regression"):
        md("### Counter-example: when re-ranking hurts\nThe cross-encoder is not always right. In this query the gold "
           "chunk was already in the baseline Top-3 and re-ranking pushed it out.")
    if True:
        md(f"### {q['id']}: {q['category']}\n**Query:** *{q['query']}*")
        code(f'show_query("{q["id"]}")')
        if q.get("observation"):
            md(f"**Observation:** {q['observation']}")

md(r"""
## 8. Latency profiling

Retrieval stages are timed over every query in the suite, repeated `N_REPEATS` times after warm-up. The LLM time is the
wall-clock time of the Gemini calls made in Section 7 (network round trip included; for cached answers this is the
latency recorded when the answer was first generated).
""")
code(r"""
N_REPEATS = 20
lat = {"embed_ms": [], "search_ms": [], "rerank_ms": []}
for _ in range(N_REPEATS):
    for q in QUERIES:
        cands, t_b = baseline_retrieve(q["query"])
        _, t_r = rerank(q["query"], cands)
        lat["embed_ms"].append(t_b["embed_ms"]); lat["search_ms"].append(t_b["search_ms"])
        lat["rerank_ms"].append(t_r["rerank_ms"])

b_total = np.array(lat["embed_ms"]) + np.array(lat["search_ms"])
r_total = b_total + np.array(lat["rerank_ms"])
llm = np.array([t for t in llm_log if t == t])

def stats(name, arr):
    return {"stage": name, "mean_ms": arr.mean(), "p50_ms": np.percentile(arr, 50), "p95_ms": np.percentile(arr, 95)}

latency_df = pd.DataFrame([
    stats("Query embedding (bi-encoder)", np.array(lat["embed_ms"])),
    stats("FAISS Top-10 search", np.array(lat["search_ms"])),
    stats("Cross-encoder re-rank (10 pairs)", np.array(lat["rerank_ms"])),
    stats("Pipeline 1 retrieval total", b_total),
    stats("Pipeline 2 retrieval total", r_total),
    stats("LLM generation (Gemini, per call)", llm),
    stats("Pipeline 1 end-to-end (retrieval + LLM)", b_total.mean() + llm),
    stats("Pipeline 2 end-to-end (retrieval + LLM)", r_total.mean() + llm),
]).round(2)
print(f"device={DEVICE}, {len(lat['embed_ms'])} timed retrievals; re-ranking overhead = "
      f"{np.mean(lat['rerank_ms']):.1f} ms ({np.mean(r_total) / np.mean(b_total):.1f}× the baseline retrieval time)")
latency_df
""")

md(r"""
## 9. Scaling experiment: why not run the cross-encoder over the whole corpus?

A bi-encoder embeds documents **once, offline**, so a query costs one forward pass plus a vector search. A cross-encoder
has nothing to pre-compute, because every (query, chunk) pair needs its own full transformer pass at query time. Below
we score **all** chunks with the cross-encoder for one query and extrapolate linearly to larger collections.
""")
code(r"""
query = QUERIES[0]["query"]
all_pairs = [(query, t) for t in texts]
_ = cross_encoder.predict(all_pairs[:64], batch_size=64)  # warm-up

t0 = time.perf_counter(); full_scores = cross_encoder.predict(all_pairs, batch_size=64); sync()
ce_full_ms = (time.perf_counter() - t0) * 1e3
t0 = time.perf_counter()
for _ in range(20): baseline_retrieve(query)
bi_ms = (time.perf_counter() - t0) * 1e3 / 20

per_pair_ms = ce_full_ms / len(texts)
embed_ms, search_ms = np.mean(lat["embed_ms"]), np.mean(lat["search_ms"])

def fmt(ms):
    if ms < 1e3: return f"{ms:,.1f} ms"
    if ms < 6e4: return f"{ms / 1e3:,.1f} s"
    return f"{ms / 6e4:,.1f} min" if ms < 3.6e6 else f"{ms / 3.6e6:,.1f} h"

# exact (flat) search grows linearly but is a cheap dot product; ANN indexes (HNSW/IVF) grow sub-linearly
scale = pd.DataFrame([{
    "collection size (chunks)": f"{n:,}",
    "bi-encoder: embed query + flat search": fmt(embed_ms + search_ms * n / len(texts)),
    "two-stage: + cross-encoder on Top-10": fmt(embed_ms + search_ms * n / len(texts) + np.mean(lat["rerank_ms"])),
    "cross-encoder over ALL chunks": fmt(per_pair_ms * n),
} for n in [len(texts), 10_000, 100_000, 1_000_000, 10_000_000]])
print(f"Cross-encoder over all {len(texts)} chunks: {ce_full_ms:.0f} ms ({per_pair_ms:.2f} ms/pair) "
      f"vs bi-encoder retrieval {bi_ms:.1f} ms")
top3_full = [label(chunks[i]) for i in np.argsort(-full_scores)[:3]]
print("Top-3 when the cross-encoder scores the full corpus:", top3_full)
scale
""")

code(r"""
# Save all measured results (used to write the report)
Path("results").mkdir(exist_ok=True)
rank_df.to_csv("results/rank_table.csv", index=False)
latency_df.to_csv("results/latency.csv", index=False)
scale.to_csv("results/scaling.csv", index=False)
json.dump({"summary": summary.to_dict(orient="records"), "per_category": per_cat.reset_index().to_dict(orient="records"),
           "answers": [{k: q.get(k) for k in ("id", "query", "truth", "answer_1", "answer_2")} for q in QUERIES if "answer_1" in q],
           "ce_full_ms": ce_full_ms, "per_pair_ms": per_pair_ms, "index_time_s": index_time, "n_chunks": len(chunks),
           "device": DEVICE, "top3_full_corpus": top3_full},
          open("results/run_results.json", "w"), indent=1, default=str)
print("saved results/")
""")

md(r"""
## 10. Summary

* Section 6 gives the aggregate effect: how often the gold chunk reaches the LLM's Top-3 context with and without re-ranking.
* Section 7 shows, query by query, how each rank shift changed the generated answer.
* Section 8 shows the price: re-ranking 10 candidates adds tens of milliseconds, small next to the LLM call.
* Section 9 shows why the cross-encoder is used only as a second stage: its cost grows linearly with the number of
  chunks and cannot be pre-computed, so it is affordable for 10 candidates but not for a whole collection.

The written analysis is in `report/report.pdf`.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "RAG_Baseline_vs_Reranking.ipynb"
nbf.write(nb, out)
print("wrote", out)
