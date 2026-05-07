"""
Stage 31: re-test law neighbor expansion (graph/law_neighbors_v1.pkl) on top of
the new multi-view best base (nL=15 nC=0, val 0.2064).

Earlier (stage 10) on the old baseline 0.1447 this was net negative. Maybe with
the stronger base it behaves differently.
"""
import os, json, pickle, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
MV_PATH  = P / "expansions/val_multiview_qwen3_14b.json"
NEIGH_PKL = P / "graph/law_neighbors_v1.pkl"

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")


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
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_PATH.read_text())
    mv = json.loads(MV_PATH.read_text())
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
    print(f"[load] neighbors: {len(neighbors):,}")

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
            v = encode(q); s, i = law_index.search(v, K_LAW_RAW)
            law_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_cache[q]
    def courts_for(q):
        if q not in court_cache:
            v = encode(q); s, i = court_index.search(v, K_COURT_RAW)
            court_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_cache[q]

    def build_row(vq):
        exp = val_exp.get(vq.query_id, {}).get("parsed") or {}
        mvp = mv.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        d = {"qid": vq.query_id, "orig": vq.query,
             "kw": " ".join(codes + kws) if (kws or codes) else None,
             "hyde": exp.get("hyde") if exp.get("hyde") and len(exp.get("hyde","")) >= 50 else None,
             "trans_concise": mvp.get("trans_concise") or None,
             "hyde_statute":  mvp.get("hyde_statute")  or None,
             "gold": split_cites(vq.gold_citations)}
        sq = mvp.get("sub_questions") or []
        d["sub_q3"] = sq[2] if len(sq) >= 3 and sq[2] and len(sq[2]) > 10 else None
        for k in list(d.keys()):
            if k in ("qid","orig","gold"): continue
            v = d.get(k)
            if not v or (isinstance(v, str) and len(v) < 20):
                d[k] = None
        return d

    rows = [build_row(vq) for vq in val.itertuples()]
    distinct = set()
    for r in rows:
        for k in MODES:
            if r.get(k): distinct.add(r[k])
    print(f"[search] pre-warm {len(distinct)}")
    for q in distinct:
        laws_for(q); courts_for(q)

    def predict(r, n_law=15, n_court=0,
                n_seed=15, w_same=0.0, w_adj=0.0):
        rs_law, rs_court = [], []
        for k in MODES:
            v = r.get(k)
            if v:
                rs_law.append(laws_for(v)); rs_court.append(courts_for(v))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        # neighbor expansion seeded from top-n_seed laws
        seeds = sorted(ls.items(), key=lambda x: -x[1])[:n_seed]
        for law, parent in seeds:
            ng = neighbors.get(law)
            if not ng: continue
            for nb in ng["same_article"]:
                ls[nb] = ls.get(nb, 0.0) + w_same * parent
            for nb in ng["adj_article"]:
                ls[nb] = ls.get(nb, 0.0) + w_adj * parent
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    golds = [r["gold"] for r in rows]

    print("\n[reference] no neighbors (matches stage 28 best)")
    preds = [predict(r) for r in rows]
    print(f"  baseline F1 = {macro_f1(preds, golds):.4f}")

    print("\n[A] (w_same, w_adj) sweep with n_seed=15, nL=15 nC=0")
    print(f"{'w_same':>8} {'w_adj':>8}    F1")
    for ws in [0.0, 0.05, 0.1, 0.2, 0.3]:
        for wa in [0.0, 0.02, 0.05, 0.1]:
            preds = [predict(r, w_same=ws, w_adj=wa) for r in rows]
            f1 = macro_f1(preds, golds)
            print(f"  {ws:>5} {wa:>5}: {f1:.4f}")

    print("\n[B] n_seed sweep with w_same=0.1 w_adj=0.05")
    for ns in [5, 10, 15, 20, 30, 50]:
        preds = [predict(r, n_seed=ns, w_same=0.1, w_adj=0.05) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  n_seed={ns}: {f1:.4f}")

    print("\n[C] best (so far) + nL/nC sweep")
    # Pick the best (ws, wa) from earlier; default keep w_same=0.1 w_adj=0.05
    for nL, nC in [(13,0),(14,0),(15,0),(15,1),(15,2),(15,3),(16,0),(17,0),(18,0)]:
        preds = [predict(r, n_law=nL, n_court=nC, w_same=0.1, w_adj=0.05) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  nL={nL} nC={nC}: {f1:.4f}")


if __name__ == "__main__":
    main()
