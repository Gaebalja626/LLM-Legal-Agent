"""
Stage 10: ablation with law-neighbor graph on val.

For each query:
  - Stage-2 baseline produces law_scores (orig+kw+hyde RRF + court→law graph)
  - From the top-N1 law candidates, expand to same_article + adj_article neighbors
  - Add neighbor candidates with weight w_same / w_adj * (parent score)
  - Take top-out_n_law, plus court_top top-out_n_court

Optionally use the new thinking-mode expansion file when present.
"""
import os, json, pickle, time, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()

EXP_BASE  = P / "expansions/val_expansions_qwen3_14b.json"
EXP_THINK = P / "expansions/val_expansions_qwen3_14b_thinking.json"
NEIGH_PKL = P / "graph/law_neighbors_v1.pkl"

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W_DEFAULT = 0.05


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]

def f1_per_query(p, g):
    p, g = set(p), set(g)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    tp = len(p & g)
    if tp == 0: return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)

def macro_f1(preds, golds):
    return float(np.mean([f1_per_query(p, g) for p, g in zip(preds, golds)]))

def rrf_merge(rankings, k=60):
    s = defaultdict(float)
    for r in rankings:
        for rank, (doc, _) in enumerate(r):
            s[doc] += 1.0 / (k + rank + 1)
    return s


def main():
    use_thinking = "--thinking" in sys.argv
    exp_path = EXP_THINK if (use_thinking and EXP_THINK.exists()) else EXP_BASE
    print(f"[load] expansions: {exp_path.name}")

    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(exp_path.read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    with open(NEIGH_PKL, "rb") as f:
        neighbors = pickle.load(f)
    print(f"  law_idx={law_index.ntotal:,}, court_idx={court_index.ntotal:,}")
    print(f"  court_to_laws={len(court_to_laws):,}, neighbors={len(neighbors):,}")

    print("[load] BGE-M3")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    enc_cache, law_cache, court_cache = {}, {}, {}
    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]
    def laws_for(q):
        if q not in law_cache:
            v = encode(q)
            s, i = law_index.search(v, K_LAW_RAW)
            law_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_cache[q]
    def courts_for(q):
        if q not in court_cache:
            v = encode(q)
            s, i = court_index.search(v, K_COURT_RAW)
            court_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_cache[q]

    rows = []
    for vq in val.itertuples():
        exp = val_exp.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde_q = exp.get("hyde") or None
        if hyde_q and len(hyde_q) < 50: hyde_q = None
        rows.append({"qid": vq.query_id, "orig": vq.query, "kw": kw_q, "hyde": hyde_q,
                     "gold": split_cites(vq.gold_citations)})

    distinct = set()
    for r in rows:
        for k in ("orig","kw","hyde"):
            if r[k]: distinct.add(r[k])
    print(f"[search] pre-warming {len(distinct)} distinct queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time() - t:.1f}s")

    def predict(r, n_law=15, n_court=10,
                n_seed=20, w_same=0.3, w_adj=0.1, graph_w=GRAPH_W_DEFAULT):
        qs = [r[k] for k in ("orig","kw","hyde") if r.get(k)]
        if not qs: qs = [r["orig"]]
        law_rk = [laws_for(q) for q in qs]
        court_rk = [courts_for(q) for q in qs]
        law_scores = rrf_merge(law_rk)
        court_scores = rrf_merge(court_rk)
        court_top = sorted(court_scores.items(), key=lambda x: -x[1])

        # court→law graph
        if graph_w > 0:
            gf = Counter()
            for d, _ in court_top[:GRAPH_POOL]:
                for law in court_to_laws.get(d, []):
                    gf[law] += 1
            for law, freq in gf.items():
                law_scores[law] = law_scores.get(law, 0.0) + freq * graph_w

        # neighbor expansion: take top-n_seed laws, add their neighbors with parent_score * w
        seeds = sorted(law_scores.items(), key=lambda x: -x[1])[:n_seed]
        for law, parent in seeds:
            ng = neighbors.get(law)
            if not ng: continue
            for nb in ng["same_article"]:
                law_scores[nb] = law_scores.get(nb, 0.0) + w_same * parent
            for nb in ng["adj_article"]:
                law_scores[nb] = law_scores.get(nb, 0.0) + w_adj * parent

        law_top = sorted(law_scores.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    golds = [r["gold"] for r in rows]

    # baseline ref
    def baseline_predict(r, n_law=15, n_court=10):
        return predict(r, n_law=n_law, n_court=n_court, n_seed=0, w_same=0, w_adj=0)

    print("\n[ref] baseline (no neighbor expansion), nL=15 nC=10")
    preds = [baseline_predict(r) for r in rows]
    print(f"  F1 = {macro_f1(preds, golds):.4f}")

    print("\n[A] neighbor weights sweep, n_seed=20, nL=15 nC=10")
    print(f"{'w_same':>7} {'w_adj':>6}    F1")
    for ws in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        for wa in [0.0, 0.05, 0.1, 0.2]:
            preds = [predict(r, w_same=ws, w_adj=wa) for r in rows]
            print(f"  {ws:>5} {wa:>5}: {macro_f1(preds, golds):.4f}")

    print("\n[B] n_seed sweep, w_same=0.3 w_adj=0.1, nL=15 nC=10")
    for ns in [5, 10, 20, 30, 50]:
        preds = [predict(r, n_seed=ns) for r in rows]
        print(f"  n_seed={ns:>2}: {macro_f1(preds, golds):.4f}")

    print("\n[C] (n_law,n_court) sweep, w_same=0.3 w_adj=0.1, n_seed=20")
    for nL in [15, 20, 25, 30, 40]:
        for nC in [10, 12, 15, 20]:
            preds = [predict(r, n_law=nL, n_court=nC) for r in rows]
            f1 = macro_f1(preds, golds)
            print(f"  nL={nL:>2} nC={nC:>2}: {f1:.4f}")


if __name__ == "__main__":
    main()
