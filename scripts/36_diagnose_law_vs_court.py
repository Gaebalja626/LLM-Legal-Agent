"""
Diagnose: how much of our F1 = 0.2064 comes from laws vs courts?

For each val query, split the gold and our prediction into laws / courts using
LAW_DOC_IDS and COURT_DOC_IDS vocabularies. Compute:

  - law-only F1   (treat only laws on both sides)
  - court-only F1 (only courts)
  - combined F1 (current sub_v6 metric)

Also: oracle ceilings — if we somehow got all retrievable laws/courts.
"""
import os, json, pickle
from pathlib import Path
from collections import defaultdict, Counter

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
    return 2 * pr * rc / (pr + rc), pr, rc, tp

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
    LAW_VOCAB = set(LAW_DOC_IDS)
    COURT_VOCAB = set(COURT_DOC_IDS)
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
    for q in distinct:
        laws_for(q); courts_for(q)

    # Best base prediction (nL=15 nC=0)
    def predict_full(r, n_law=15, n_court=0):
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
        law_top = sorted(ls.items(), key=lambda x: -x[1])
        return ([d for d, _ in law_top[:n_law]] + [d for d, _ in court_top[:n_court]],
                law_top, court_top)

    def is_law(c):
        return c in LAW_VOCAB
    def is_court(c):
        return c in COURT_VOCAB

    def split_law_court(citations):
        laws  = [c for c in citations if is_law(c)]
        courts = [c for c in citations if is_court(c)]
        return laws, courts

    print("Per-query breakdown (best config nL=15 nC=0):")
    print(f"  {'qid':<8} {'gold(L|C)':>10} {'pred_L':>7} {'tp_L':>5} {'P_L':>5} {'R_L':>5} {'F1_L':>6}  {'F1_full':>7}")

    sum_f1_law, sum_f1_full = 0, 0
    n = 0
    for r in rows:
        gold = r["gold"]
        pred, law_top, court_top = predict_full(r)
        gold_l, gold_c = split_law_court(gold)
        pred_l, pred_c = split_law_court(pred)

        # law-only F1
        gset = set(gold_l); pset = set(pred_l)
        if not gset and not pset:
            f1_l = 1.0; pr = 1.0; rc = 1.0; tp_l = 0
        elif not gset or not pset:
            f1_l = 0.0; pr = 0.0; rc = 0.0
            tp_l = len(gset & pset)
        else:
            tp_l = len(gset & pset)
            pr = tp_l / len(pset) if pset else 0.0
            rc = tp_l / len(gset) if gset else 0.0
            f1_l = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.0

        # full F1 (current metric)
        gset_f = set(gold); pset_f = set(pred)
        if not gset_f and not pset_f:
            f1_full = 1.0
        elif not gset_f or not pset_f:
            f1_full = 0.0
        else:
            tp = len(gset_f & pset_f)
            prf = tp / len(pset_f); rcf = tp / len(gset_f)
            f1_full = 2*prf*rcf/(prf+rcf) if (prf+rcf) > 0 else 0.0

        sum_f1_law += f1_l; sum_f1_full += f1_full; n += 1
        print(f"  {r['qid']:<8} {len(gold_l):>4}|{len(gold_c):<4} {len(pred_l):>7} {tp_l:>5} {pr:>5.2f} {rc:>5.2f} {f1_l:>6.3f}  {f1_full:>7.3f}")

    print(f"\nMacro F1 (law-only): {sum_f1_law/n:.4f}")
    print(f"Macro F1 (full):     {sum_f1_full/n:.4f}  ← current metric")

    # Court oracle: how many gold courts even appear in our retrieval pool?
    print("\n--- Court retrieval ceiling per query ---")
    print(f"  {'qid':<8} {'gold_C':>6} {'in_top200_RRF':>14}")
    for r in rows:
        gold_l, gold_c = split_law_court(r["gold"])
        _, _, court_top = predict_full(r)
        court_set = {d for d, _ in court_top[:200]}
        hits = sum(1 for c in gold_c if c in court_set)
        print(f"  {r['qid']:<8} {len(gold_c):>6} {hits:>10}/{len(gold_c)}  rate={hits/max(1,len(gold_c)):.2f}")

    # If we could perfectly pick courts (oracle), what F1?
    print("\n--- Oracle: assume we output ALL gold courts (perfect court precision) ---")
    sum_oracle = 0
    for r in rows:
        gold = r["gold"]
        pred, law_top, court_top = predict_full(r)
        gold_l, gold_c = split_law_court(gold)
        pred_l, pred_c = split_law_court(pred)
        # final = our laws + ALL gold courts
        ideal = pred_l + gold_c
        tp = len(set(ideal) & set(gold))
        prf = tp / len(ideal) if ideal else 0
        rcf = tp / len(gold) if gold else 0
        f1_oracle = 2*prf*rcf/(prf+rcf) if (prf+rcf) > 0 else 0
        sum_oracle += f1_oracle
        print(f"  {r['qid']}: pred_L={len(pred_l)} gold_C={len(gold_c)} → F1_with_oracle_courts={f1_oracle:.3f}")
    print(f"Macro F1 with oracle courts: {sum_oracle/n:.4f}")


if __name__ == "__main__":
    main()
