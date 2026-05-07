"""
Stage 28: finer size sweep around the new optimum (nL=15 nC=3 gw=0.05 → 0.1895).

Specifically tries small nC values (1, 2) which were absent before.
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

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
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
    print(f"[search] pre-warming {len(distinct)} queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    def stage2_scores(r, graph_w):
        rs_law, rs_court = [], []
        for k in MODES:
            v = r.get(k)
            if v:
                rs_law.append(laws_for(v))
                rs_court.append(courts_for(v))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        if graph_w > 0:
            gf = Counter()
            for d, _ in court_top[:GRAPH_POOL]:
                for law in court_to_laws.get(d, []):
                    gf[law] += 1
            for law, freq in gf.items():
                ls[law] = ls.get(law, 0.0) + freq * graph_w
        law_top = sorted(ls.items(), key=lambda x: -x[1])
        return law_top, court_top

    golds = [r["gold"] for r in rows]
    nLs = [11, 12, 13, 14, 15, 16, 17, 18]
    nCs = [0, 1, 2, 3, 4, 5]
    gws = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

    results = []
    for gw in gws:
        rankings = [stage2_scores(r, gw) for r in rows]
        for nL in nLs:
            for nC in nCs:
                preds = []
                for (law_top, court_top) in rankings:
                    preds.append([d for d,_ in law_top[:nL]] + [d for d,_ in court_top[:nC]])
                f1 = macro_f1(preds, golds)
                results.append((f1, nL, nC, gw, preds))

    results.sort(key=lambda x: -x[0])

    print(f"\n[top 20 configs]")
    print(f"  {'F1':>7}  {'nL':>3} {'nC':>3} {'gw':>5}")
    for f1, nL, nC, gw, _ in results[:20]:
        print(f"  {f1:>7.4f}  {nL:>3} {nC:>3} {gw:>5}")

    print(f"\n[best per nC]")
    best_per_nc = {}
    for f1, nL, nC, gw, _ in results:
        if nC not in best_per_nc or f1 > best_per_nc[nC][0]:
            best_per_nc[nC] = (f1, nL, gw)
    for nC in sorted(best_per_nc.keys()):
        f1, nL, gw = best_per_nc[nC]
        print(f"  nC={nC}: best F1={f1:.4f} (nL={nL} gw={gw})")

    print(f"\n[per-query for top 3]")
    for f1, nL, nC, gw, preds in results[:3]:
        per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
        print(f"  nL={nL} nC={nC} gw={gw} F1={f1:.4f}  {per_q}")


if __name__ == "__main__":
    main()
