"""
Stage 24: deeper combo ablation building on stage-23 winners.

Stage-23 single-view winners over baseline (orig+kw+hyde):
  + hyde_statute   : 0.1502  (+0.0055)
  + sub_q3         : 0.1499  (+0.0052)

This stage tests:
  - 2-view additions (e.g. baseline + hyde_statute + sub_q3)
  - 3-view additions
  - swap one of orig/kw/hyde for hyde_statute (replacement, not addition)
  - graph_w / size sweep on top of the best combo
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
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde_q = exp.get("hyde") or None
        if hyde_q and len(hyde_q) < 50: hyde_q = None
        d = {
            "qid": vq.query_id,
            "orig": vq.query,
            "kw": kw_q,
            "hyde": hyde_q,
            "gold": split_cites(vq.gold_citations),
            "trans_natural":  mvp.get("trans_natural") or None,
            "trans_formal":   mvp.get("trans_formal")  or None,
            "trans_concise":  mvp.get("trans_concise") or None,
            "hyde_statute":   mvp.get("hyde_statute")  or None,
            "hyde_case":      mvp.get("hyde_case")     or None,
        }
        sq = mvp.get("sub_questions") or []
        for i, q in enumerate(sq[:3], 1):
            d[f"sub_q{i}"] = q if q and len(q) > 10 else None
        for k in list(d.keys()):
            if k in ("qid","orig","gold"): continue
            v = d.get(k)
            if not v or (isinstance(v, str) and len(v) < 20):
                d[k] = None
        return d

    rows = [build_row(vq) for vq in val.itertuples()]
    all_keys = ["orig","kw","hyde","trans_natural","trans_formal","trans_concise",
                "hyde_statute","hyde_case","sub_q1","sub_q2","sub_q3"]
    distinct = set()
    for r in rows:
        for k in all_keys:
            if r.get(k): distinct.add(r[k])
    print(f"[search] pre-warming {len(distinct)} distinct queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    def predict(r, modes, n_law=15, n_court=10, graph_w=0.05):
        rs_law, rs_court = [], []
        for k in modes:
            v = r.get(k)
            if v:
                rs_law.append(laws_for(v))
                rs_court.append(courts_for(v))
        if not rs_law:
            rs_law.append(laws_for(r["orig"]))
            rs_court.append(courts_for(r["orig"]))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        if graph_w > 0:
            gf = Counter()
            for d, _ in court_top[:GRAPH_POOL]:
                for law in court_to_laws.get(d, []):
                    gf[law] += 1
            for law, freq in gf.items():
                ls[law] = ls.get(law, 0.0) + freq * graph_w
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    golds = [r["gold"] for r in rows]
    base = ("orig","kw","hyde")

    def fmt(modes):
        return "+".join(modes)

    print("\n[A] baseline + (hyde_statute, X) — try every other view as 2nd add")
    for second in ["hyde_case","sub_q1","sub_q2","sub_q3","trans_natural","trans_formal","trans_concise"]:
        modes = base + ("hyde_statute", second)
        preds = [predict(r, modes) for r in rows]
        f1 = macro_f1(preds, golds)
        per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
        print(f"  {fmt(modes):<60}: {f1:.4f}  {per_q}")

    print("\n[B] baseline + (sub_q3, X)")
    for second in ["hyde_statute","hyde_case","sub_q1","sub_q2","trans_natural","trans_formal","trans_concise"]:
        modes = base + ("sub_q3", second)
        preds = [predict(r, modes) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  {fmt(modes):<60}: {f1:.4f}")

    print("\n[C] baseline + 3 view triples (combinations of statute/case/sub_q3/concise)")
    triples = [
        ("hyde_statute","hyde_case","sub_q3"),
        ("hyde_statute","sub_q3","trans_concise"),
        ("hyde_statute","sub_q3","sub_q1"),
        ("hyde_statute","sub_q3","sub_q2"),
        ("hyde_statute","hyde_case","trans_concise"),
        ("hyde_statute","sub_q1","sub_q3"),
        ("sub_q1","sub_q2","sub_q3"),
        ("hyde_statute","sub_q3","hyde_case"),
    ]
    for trio in triples:
        modes = base + trio
        preds = [predict(r, modes) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  {fmt(modes):<70}: {f1:.4f}")

    print("\n[D] swap experiments — replace one of orig/kw/hyde with hyde_statute")
    swaps = [
        ("orig","kw","hyde_statute"),
        ("orig","hyde_statute","hyde"),
        ("hyde_statute","kw","hyde"),
        ("kw","hyde","hyde_statute"),
        ("orig","kw","hyde","hyde_statute"),  # for reference (= add)
    ]
    for modes in swaps:
        preds = [predict(r, modes) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  {fmt(modes):<60}: {f1:.4f}")

    print("\n[E] best 2-add (baseline + hyde_statute + sub_q3) — size + graph_w sweep")
    best = base + ("hyde_statute", "sub_q3")
    for nL, nC in [(15,10),(20,10),(20,12),(25,15),(25,10),(30,10)]:
        preds = [predict(r, best, n_law=nL, n_court=nC) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  size nL={nL} nC={nC}: {f1:.4f}")
    print()
    for gw in [0.0, 0.03, 0.05, 0.08, 0.1, 0.15]:
        preds = [predict(r, best, graph_w=gw) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  graph_w={gw}: {f1:.4f}")


if __name__ == "__main__":
    main()
