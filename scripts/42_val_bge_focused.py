"""
Ablation: BGE-focused retrieval (A + C).

Adds two new signals on top of the multi-view baseline (val 0.2064):

  A. BGE-only direct retrieval:
     for each query variant, search bge_only_v2.faiss top-K_bge → BGE rankings
     fuse via RRF into court_scores

  C. court → cited BGE graph 1-hop:
     for each top-pool retrieved court, look up court_to_bges_v1.pkl
     boost cited BGEs in court_scores (similar to court→law graph for laws)

Test combinations and sweep output sizes.
"""
import os, json, pickle, time, re
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
BGE_FAISS = P / "indexes/bge_only_v2.faiss"
BGE_DOC   = P / "indexes/bge_only_doc_ids_v2.pkl"
COURT_TO_BGES = P / "graph/court_to_bges_v1.pkl"

K_LAW = 200
K_COURT = 200
K_BGE = 200
GRAPH_POOL = 100
GRAPH_W = 0.05
BGE_GRAPH_POOL = 100
BGE_GRAPH_W = 0.05
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
    with open(BGE_DOC, "rb") as f:
        BGE_DOC_IDS = pickle.load(f)
    BGE_VOCAB = set(BGE_DOC_IDS)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    bge_index = faiss.read_index(str(BGE_FAISS))
    print(f"[idx] law={law_index.ntotal:,} court={court_index.ntotal:,} bge={bge_index.ntotal:,}")
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    if COURT_TO_BGES.exists():
        with open(COURT_TO_BGES, "rb") as f:
            court_to_bges = pickle.load(f)
        print(f"[graph] court_to_bges: {len(court_to_bges):,}")
    else:
        court_to_bges = {}
        print(f"[graph] court_to_bges: NOT BUILT YET ({COURT_TO_BGES})")

    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256
    enc_cache, law_cache, court_cache, bge_cache = {}, {}, {}, {}
    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]
    def laws_for(q):
        if q not in law_cache:
            v = encode(q); s, i = law_index.search(v, K_LAW)
            law_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_cache[q]
    def courts_for(q):
        if q not in court_cache:
            v = encode(q); s, i = court_index.search(v, K_COURT)
            court_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_cache[q]
    def bges_for(q):
        if q not in bge_cache:
            v = encode(q); s, i = bge_index.search(v, K_BGE)
            bge_cache[q] = [(BGE_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return bge_cache[q]

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
    print(f"[search] pre-warm {len(distinct)} on 3 indexes")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q); bges_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    def predict(r,
                use_A=False, use_C=False,
                bge_w_in_court_rrf=1.0,    # how heavily to weight BGE retrieval rankings vs courts
                bge_graph_w=0.05,
                n_law=15, n_court=0, n_bge=0,
                graph_w=0.05):
        # law rankings
        rs_law = []
        for k in MODES:
            v = r.get(k)
            if v: rs_law.append(laws_for(v))
        # court rankings (full court_index)
        rs_court = []
        for k in MODES:
            v = r.get(k)
            if v: rs_court.append(courts_for(v))
        # A: BGE-only rankings, fused into court_scores (BGE은 court의 부분집합)
        rs_bge = []
        if use_A:
            for k in MODES:
                v = r.get(k)
                if v: rs_bge.append(bges_for(v))

        # Build scores
        ls = rrf_merge(rs_law)
        cs = rrf_merge(rs_court)
        bs = rrf_merge(rs_bge) if rs_bge else {}

        # Court-side: fuse cs + bs (BGE takes precedence by weight)
        # Approach: compute bs separately, then add bs * weight into cs
        for d, sc in bs.items():
            cs[d] = cs.get(d, 0.0) + bge_w_in_court_rrf * sc

        court_top = sorted(cs.items(), key=lambda x: -x[1])

        # Court→Law graph 1-hop boost on laws
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * graph_w

        # C: court→BGE graph 1-hop, boost BGE in court_scores
        if use_C and court_to_bges:
            bge_freq = Counter()
            for d, _ in court_top[:BGE_GRAPH_POOL]:
                for bge in court_to_bges.get(d, []):
                    bge_freq[bge] += 1
            for bge, freq in bge_freq.items():
                cs[bge] = cs.get(bge, 0.0) + freq * bge_graph_w
            court_top = sorted(cs.items(), key=lambda x: -x[1])

        law_top = sorted(ls.items(), key=lambda x: -x[1])[:n_law]

        out = [d for d, _ in law_top]
        # Append courts/BGEs by quota
        if n_court > 0:
            out += [d for d, _ in court_top[:n_court]]
        # Optionally explicit BGE-only quota (separate from general court_top)
        if n_bge > 0 and bs:
            bge_only_ranked = sorted(bs.items(), key=lambda x: -x[1])
            seen = set(out)
            for d, _ in bge_only_ranked:
                if d not in seen:
                    seen.add(d); out.append(d)
                    if sum(1 for c in out if c in BGE_VOCAB) >= n_bge:
                        break
        return out

    golds = [r["gold"] for r in rows]

    print("\n[ref] baseline (no A no C, nL=15 nC=0)")
    print(f"  F1 = {macro_f1([predict(r) for r in rows], golds):.4f}")

    print("\n[A only] add BGE-only ranking into court_scores")
    print(f"{'bge_w':>6} {'nL':>3} {'nC':>3}     F1")
    for bge_w in [0.5, 1.0, 2.0]:
        for nL, nC in [(15,0),(15,3),(15,5),(15,10),(12,3),(13,5)]:
            preds = [predict(r, use_A=True, bge_w_in_court_rrf=bge_w,
                             n_law=nL, n_court=nC) for r in rows]
            f1 = macro_f1(preds, golds)
            print(f"  {bge_w:>5} {nL:>3} {nC:>3}: {f1:.4f}")

    if not court_to_bges:
        print("\n[C / A+C] court_to_bges not built yet — skip")
        return

    print("\n[C only] court→BGE graph boost")
    print(f"{'bge_gw':>7} {'nL':>3} {'nC':>3}     F1")
    for gw in [0.02, 0.05, 0.1, 0.2]:
        for nL, nC in [(15,0),(15,3),(15,5),(15,10)]:
            preds = [predict(r, use_C=True, bge_graph_w=gw,
                             n_law=nL, n_court=nC) for r in rows]
            f1 = macro_f1(preds, golds)
            print(f"  {gw:>5} {nL:>3} {nC:>3}: {f1:.4f}")

    print("\n[A+C combined]")
    print(f"{'bge_w':>6} {'bge_gw':>7} {'nL':>3} {'nC':>3}     F1")
    for bge_w in [1.0, 2.0]:
        for gw in [0.05, 0.1]:
            for nL, nC in [(15,0),(15,3),(15,5),(15,10),(13,5),(12,8)]:
                preds = [predict(r, use_A=True, use_C=True,
                                 bge_w_in_court_rrf=bge_w, bge_graph_w=gw,
                                 n_law=nL, n_court=nC) for r in rows]
                f1 = macro_f1(preds, golds)
                print(f"  {bge_w:>5} {gw:>5} {nL:>3} {nC:>3}: {f1:.4f}")


if __name__ == "__main__":
    main()
