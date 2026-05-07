"""
Stage 19: add BM25 v2 (German corpus) as a 4th ranking in RRF.

Hypothesis: kw_q (= "<codes> <German keywords>") is fully German, so BM25 over
the German corpus has signal we discarded earlier.

For each query variant we run BGE-M3 dense (split). Additionally we run BM25
on kw_q (and optionally hyde) and split its results into laws/courts using the
LAW_VOCAB / COURT_VOCAB. Then RRF over all rankings + graph boost as before.

Compare F1 to the current best (0.1447).
"""
import os, json, pickle, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import bm25s

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"

K_LAW_RAW, K_COURT_RAW = 200, 200
K_BM25 = 400
GRAPH_POOL = 100
GRAPH_W = 0.05


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
    print("[load] data + indexes")
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_PATH.read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    LAW_VOCAB = set(LAW_DOC_IDS)
    COURT_VOCAB = set(COURT_DOC_IDS)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)

    print("[load] BM25 v2")
    bm25_idx = bm25s.BM25.load(str(P / "indexes/bm25_v2"), load_corpus=False)
    with open(P / "indexes/bm25_v2/doc_ids.pkl", "rb") as f:
        BM25_DOC_IDS = pickle.load(f)
    print(f"  bm25 docs: {len(BM25_DOC_IDS):,}")

    print("[load] BGE-M3")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    enc_cache, law_cache, court_cache, bm25_cache = {}, {}, {}, {}
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

    def bm25_for(q):
        """Return (laws_ranking, courts_ranking) split from BM25 top-K."""
        if q in bm25_cache:
            return bm25_cache[q]
        toks = bm25s.tokenize([q], show_progress=False)
        results, scores = bm25_idx.retrieve(toks, k=K_BM25, show_progress=False, n_threads=4)
        ids = results[0]
        sc = scores[0]
        laws, courts = [], []
        for j, s in zip(ids, sc):
            cit = BM25_DOC_IDS[int(j)]
            if cit in LAW_VOCAB:
                laws.append((cit, float(s)))
            elif cit in COURT_VOCAB:
                courts.append((cit, float(s)))
        bm25_cache[q] = (laws, courts)
        return laws, courts

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
    print(f"[search] pre-warm {len(distinct)} dense queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    # BM25 warm: only kw + hyde (German), not orig (English)
    bm25_qs = set()
    for r in rows:
        if r["kw"]: bm25_qs.add(r["kw"])
        if r["hyde"]: bm25_qs.add(r["hyde"])
    print(f"[bm25] pre-warm {len(bm25_qs)} German queries")
    t = time.time()
    for q in bm25_qs:
        bm25_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    # Quick recall sanity on val_010 (most broken)
    print("\n[sanity] val_010 BM25 hits per variant")
    for r in rows:
        if r["qid"] != "val_010": continue
        for label in ("kw", "hyde"):
            if not r[label]: continue
            bml, bmc = bm25_for(r[label])
            n_law_hit = sum(1 for d, _ in bml if d in r["gold"])
            n_court_hit = sum(1 for d, _ in bmc if d in r["gold"])
            print(f"  {label}: {n_law_hit} law hits, {n_court_hit} court hits in BM25 top-{K_BM25}")
        break

    def predict(r, n_law=15, n_court=10, use_bm25_kw=True, use_bm25_hyde=False):
        rs_law, rs_court = [], []
        for k in ("orig","kw","hyde"):
            if r.get(k):
                rs_law.append(laws_for(r[k]))
                rs_court.append(courts_for(r[k]))
        if use_bm25_kw and r.get("kw"):
            bml, bmc = bm25_for(r["kw"])
            rs_law.append(bml); rs_court.append(bmc)
        if use_bm25_hyde and r.get("hyde"):
            bml, bmc = bm25_for(r["hyde"])
            rs_law.append(bml); rs_court.append(bmc)
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    golds = [r["gold"] for r in rows]

    print("\n[ablation] BM25 add modes (nL=15 nC=10)")
    for label, kw_, hy_ in [
        ("baseline (no BM25)",  False, False),
        ("BM25(kw)",            True,  False),
        ("BM25(kw+hyde)",       True,  True),
        ("BM25(hyde)",          False, True),
    ]:
        preds = [predict(r, use_bm25_kw=kw_, use_bm25_hyde=hy_) for r in rows]
        f1 = macro_f1(preds, golds)
        per_q = {qid: round(f1_per_query(p, g), 3) for qid, p, g in zip([r["qid"] for r in rows], preds, golds)}
        print(f"  {label:<24}: F1={f1:.4f}  per_q={[per_q[q] for q in val['query_id']]}")

    print("\n[ablation] best BM25 + (n_law,n_court) sweep")
    for nL, nC in [(15,10),(20,10),(20,12),(25,15),(25,10),(30,15)]:
        preds = [predict(r, n_law=nL, n_court=nC, use_bm25_kw=True, use_bm25_hyde=False) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  BM25(kw) nL={nL:>2} nC={nC:>2}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
