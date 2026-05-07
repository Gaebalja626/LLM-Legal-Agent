"""
Stage 29: Plan G ablation re-run with the new best base config (nL=15 nC=0).

Reuses the cached verification JSON from stage-26 (val_verify_reasoning_qwen3_14b.json)
so we don't re-run the LLM. Just rebuilds the base ranking with the new params.
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
VERIFY_PATH = P / "expansions/val_verify_reasoning_qwen3_14b.json"

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
    verify = json.loads(VERIFY_PATH.read_text())

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
    print(f"[search] pre-warming {len(distinct)}")
    for q in distinct:
        laws_for(q); courts_for(q)

    def stage2(r):
        rs_law, rs_court = [], []
        for k in MODES:
            v = r.get(k)
            if v:
                rs_law.append(laws_for(v))
                rs_court.append(courts_for(v))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        law_top = sorted(ls.items(), key=lambda x: -x[1])
        return law_top, court_top

    rankings = [stage2(r) for r in rows]
    golds = [r["gold"] for r in rows]

    def variant(r, idx, kind, nL, nC, target_size):
        law_top, court_top = rankings[idx]
        base = [d for d, _ in law_top[:nL]] + [d for d, _ in court_top[:nC]]
        v = (verify.get(r["qid"]) or {}).get("parsed") or {}
        matched = list(v.get("matched", []) or [])
        additional = list(v.get("additional", []) or [])
        if kind == "baseline":
            return base
        if kind == "base+add":
            seen = set(); out = []
            for c in base + additional:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]
        if kind == "matched+add":
            seen = set(); out = []
            for c in matched + additional:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]
        if kind == "base_filter_keep_matched_then_add":
            keep = [c for c in base if c in set(matched)]
            seen = set(keep); out = list(keep)
            for c in additional:
                if c not in seen:
                    seen.add(c); out.append(c)
            for c in base:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]

    print("\n=== Plan G with new baseline (nL=15 nC=0 gw=0.05) ===")
    base_nL, base_nC = 15, 0
    print(f"baseline only ({base_nL}+{base_nC}={base_nL+base_nC}): "
          f"{macro_f1([variant(r, i, 'baseline', base_nL, base_nC, 25) for i, r in enumerate(rows)], golds):.4f}")

    for kind in ["base+add", "matched+add", "base_filter_keep_matched_then_add"]:
        print(f"\n  [{kind}] target_size sweep:")
        for ts in [12, 13, 14, 15, 16, 17, 18, 20, 22, 25]:
            preds = [variant(r, i, kind, base_nL, base_nC, ts) for i, r in enumerate(rows)]
            f1 = macro_f1(preds, golds)
            per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
            print(f"    target={ts}: F1={f1:.4f}")

    # Also try base = nL=15 nC=3 (top size sweep before nC=0)
    print(f"\n=== Plan G with nL=15 nC=3 gw=0.05 base ===")
    base_nL, base_nC = 15, 3
    print(f"baseline only ({base_nL}+{base_nC}={base_nL+base_nC}): "
          f"{macro_f1([variant(r, i, 'baseline', base_nL, base_nC, 25) for i, r in enumerate(rows)], golds):.4f}")

    for kind in ["base+add", "base_filter_keep_matched_then_add"]:
        print(f"\n  [{kind}] target_size sweep:")
        for ts in [15, 16, 17, 18, 20, 22, 25, 28]:
            preds = [variant(r, i, kind, base_nL, base_nC, ts) for i, r in enumerate(rows)]
            f1 = macro_f1(preds, golds)
            print(f"    target={ts}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
