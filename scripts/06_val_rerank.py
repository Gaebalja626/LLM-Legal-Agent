"""
Stage 6: cross-encoder rerank on top of agentic retrieval.

For each val query:
  1. Get top-K_in_law law candidates and top-K_in_court court candidates
     from the existing pipeline (orig+kw+hyde RRF + graph boost).
  2. Score (rerank_query, candidate_text) pairs with BGE-reranker-v2-m3.
  3. Sort by reranker score, take top-N_law + top-N_court.
  4. Report Macro F1 across rerank-query choice and N sweeps.

Notes:
  - Reranker query options: 'orig' (English) and 'hyde' (German narrative).
  - max_length=512 truncates long court texts; that is the published default
    for BGE-reranker-v2-m3.
"""
import os, json, pickle, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
CIT_PKL  = P / "parquet/citation_text_v2.pkl"

# Stage-2 retrieval config (fixed to val-best for now)
K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]


def f1_per_query(pred, gold):
    p, g = set(pred), set(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    tp = len(p & g)
    if tp == 0: return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def macro_f1(preds, golds):
    return float(np.mean([f1_per_query(p, g) for p, g in zip(preds, golds)]))


def rrf_merge(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking):
            scores[doc] += 1.0 / (k + rank + 1)
    return scores


def main():
    print("[load] resources")
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_PATH.read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)
    print(f"  cit2text entries: {len(cit2text):,}")

    print("[load] BGE-M3 (encoder)")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    # Caches
    enc_cache = {}
    law_cache = {}
    court_cache = {}

    def encode_q(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]

    def laws_for(q):
        if q not in law_cache:
            v = encode_q(q)
            s, i = law_index.search(v, K_LAW_RAW)
            law_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_cache[q]

    def courts_for(q):
        if q not in court_cache:
            v = encode_q(q)
            s, i = court_index.search(v, K_COURT_RAW)
            court_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_cache[q]

    # Build candidate pools per query (law top-K_in_law, court top-K_in_court)
    rows = []
    for vq in val.itertuples():
        exp = val_exp.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde_q = (exp.get("hyde") or "")
        if len(hyde_q) < 50:
            hyde_q = None
        rows.append({
            "qid": vq.query_id,
            "orig": vq.query,
            "kw": kw_q,
            "hyde": hyde_q,
            "gold": split_cites(vq.gold_citations),
        })

    distinct = set()
    for r in rows:
        for k in ("orig", "kw", "hyde"):
            if r[k]:
                distinct.add(r[k])
    print(f"[search] pre-warming {len(distinct)} distinct queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time() - t:.1f}s")

    def candidates(r, K_in_law=100, K_in_court=100):
        qs = [r[k] for k in ("orig", "kw", "hyde") if r.get(k)]
        if not qs:
            qs = [r["orig"]]
        law_rankings = [laws_for(q) for q in qs]
        court_rankings = [courts_for(q) for q in qs]
        law_scores = rrf_merge(law_rankings)
        court_scores = rrf_merge(court_rankings)

        court_ranked = sorted(court_scores.items(), key=lambda x: -x[1])
        # Graph boost on laws using top court pool
        graph_freq = Counter()
        for d, _ in court_ranked[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                graph_freq[law] += 1
        for law, freq in graph_freq.items():
            law_scores[law] = law_scores.get(law, 0.0) + freq * GRAPH_W
        law_ranked = sorted(law_scores.items(), key=lambda x: -x[1])

        return [d for d, _ in law_ranked[:K_in_law]], [d for d, _ in court_ranked[:K_in_court]]

    print("[load] BGE-reranker-v2-m3 (cross-encoder)")
    rerank = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")

    def rerank_pairs(query, cits, batch_size=32):
        if not cits:
            return []
        texts = [cit2text.get(c, c) for c in cits]
        pairs = [(query, t) for t in texts]
        scores = rerank.predict(pairs, batch_size=batch_size,
                                show_progress_bar=False, convert_to_numpy=True)
        return list(zip(cits, scores.tolist()))

    def predict(r, rerank_query="orig", K_in_law=100, K_in_court=100,
                n_law=15, n_court=10):
        law_cits, court_cits = candidates(r, K_in_law=K_in_law, K_in_court=K_in_court)
        rq = r[rerank_query] or r["orig"]
        law_scored = rerank_pairs(rq, law_cits)
        court_scored = rerank_pairs(rq, court_cits)
        law_top = sorted(law_scored, key=lambda x: -x[1])[:n_law]
        court_top = sorted(court_scored, key=lambda x: -x[1])[:n_court]
        return [d for d, _ in law_top] + [d for d, _ in court_top]

    golds = [r["gold"] for r in rows]

    print("\n[ablation A] rerank_query (K_in=100, n_law=15, n_court=10)")
    for rq in ("orig", "hyde"):
        t = time.time()
        preds = [predict(r, rerank_query=rq) for r in rows]
        f1 = macro_f1(preds, golds)
        per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
        print(f"  rq={rq:<5}: F1={f1:.4f}  per_q={per_q}  ({time.time()-t:.1f}s)")

    # Use best rerank_query for next ablations (we'll discover from above)
    # but compute both and keep results — easier to just iterate config
    print("\n[ablation B] (n_law, n_court) sweep, rq=orig, K_in=100")
    for nL, nC in [(15, 10), (20, 12), (25, 15), (30, 15), (30, 20), (40, 20)]:
        preds = [predict(r, rerank_query="orig", n_law=nL, n_court=nC) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  nL={nL:>2} nC={nC:>2}: F1={f1:.4f}")

    print("\n[ablation C] K_in (candidate pool) sweep, rq=orig, n_law=15 n_court=10")
    for kL, kC in [(50, 50), (100, 100), (200, 200), (300, 200)]:
        preds = [predict(r, rerank_query="orig", K_in_law=kL, K_in_court=kC) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  K_in_law={kL} K_in_court={kC}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
