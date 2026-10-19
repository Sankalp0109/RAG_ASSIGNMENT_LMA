"""Scratch screening: run candidate queries through bi-encoder + cross-encoder and print gold ranks."""
import sys, json
sys.path.insert(0, "src")
import numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from chunking import chunk_document

chunks = chunk_document("corpus_document.pdf")
bi = SentenceTransformer(sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2")
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
emb = bi.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=64)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype("float32"))

Q = json.load(open("src/candidates.json"))
for q in Q:
    qv = bi.encode([q["q"]], normalize_embeddings=True).astype("float32")
    s, I = index.search(qv, 10)
    cand = list(I[0])
    sc = ce.predict([(q["q"], chunks[i]["text"]) for i in cand])
    order = [cand[j] for j in np.argsort(-sc)]
    norm = lambda t: " ".join(t.split()).lower()
    is_gold = lambda i: any(norm(g) in norm(chunks[i]["text"]) for g in q["g"])
    allg = [c["chunk_id"] for c in chunks if is_gold(c["chunk_id"])]
    if not allg: print("!! no gold chunk for", q["q"]); continue
    br = [r + 1 for r, i in enumerate(cand) if is_gold(i)]
    cr = [r + 1 for r, i in enumerate(order) if is_gold(i)]
    flag = "WIN " if (br and cr and min(cr) <= 3 < min(br)) else ("LOSS" if (br and cr and min(br) <= 3 < min(cr)) else "    ")
    print(f"{flag} base={br} ce={cr} | {q['q']}")
    if "-v" in sys.argv:
        print("   base:", [(chunks[i]["subs"] or [chunks[i]["article"]])[0] for i in cand])
        print("   ce  :", [(chunks[i]["subs"] or [chunks[i]["article"]])[0] for i in order])
