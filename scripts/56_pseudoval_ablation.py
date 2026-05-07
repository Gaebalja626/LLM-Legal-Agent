"""
Ablation on pseudo-val + val for our LB-relevant configs.

For each config in CONFIGS, evaluate:
  - val (10) Macro F1
  - pseudo-val (100) Macro F1
  - combined (110) Macro F1

Configs include:
  sub_v6 baseline (oC0, no plan G, no graph C)
  sub_v7 plan G   (oC0 t=17 +planG, our LB best)
  sub_v8 combined (oC3 t=20 +graphC +planG)
  variants
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

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
BGE_GRAPH_POOL = 100
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")

CONFIGS = [
    # name, n_law, n_court, target_size, use_graph_c, use_plan_g, graph_w, bge_graph_w
    ("v6_oC0_t15",                      15, 0,  15, False, False, 0.05, 0.05),  # multi-view alone
    ("v6_oC3_t18",                      15, 3,  18, False, False, 0.05, 0.05),
    ("v7_oC0_t17_pG",                   15, 0,  17, False, True,  0.05, 0.05),  # LB best
    ("v8_oC3_t18_gC",                   15, 3,  18, True,  False, 0.05, 0.05),  # graph C only
    ("v8_oC3_t20_gC",                   15, 3,  20, True,  False, 0.05, 0.05),
    ("v8_oC3_t20_gC_pG",                15, 3,  20, True,  True,  0.05, 0.05),  # combined
    ("v8_oC3_t18_gC_pG",                15, 3,  18, True,  True,  0.05, 0.05),
    ("v8_oC0_t17_gC_pG",                15, 0,  17, True,  True,  0.05, 0.05),
    ("v8_oC0_t17_pG_no_gC",             15, 0,  17, False, True,  0.05, 0.05),
    ("v8_oC3_t22_gC_pG",                15, 3,  22, True,  True,  0.05, 0.05),
    ("v6_oC2_t17",                      15, 2,  17, False, False, 0.05, 0.05),
    ("v6_oC2_t17_pG",                   15, 2,  17, False, True,  0.05, 0.05),
    ("v8_oC2_t19_gC_pG",                15, 2,  19, True,  True,  0.05, 0.05),
    ("v8_oC4_t21_gC_pG",                15, 4,  21, True,  True,  0.05, 0.05),
    ("v8_oC3_t20_gC_pG_bge_w_high",     15, 3,  20, True,  True,  0.05, 0.10),
]


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


def build_row(qid, query, exp_r1, mvp):
    kws = (exp_r1 or {}).get("keywords_de") or []
    codes = (exp_r1 or {}).get("law_codes") or []
    hyde = (exp_r1 or {}).get("hyde") or ""
    d = {"qid": qid, "orig": query,
         "kw": " ".join(codes + kws) if (kws or codes) else None,
         "hyde": hyde if hyde and len(hyde) >= 50 else None,
         "trans_concise": (mvp or {}).get("trans_concise") or None,
         "hyde_statute":  (mvp or {}).get("hyde_statute") or None}
    sq = (mvp or {}).get("sub_questions") or []
    d["sub_q3"] = sq[2] if len(sq) >= 3 and sq[2] and len(sq[2]) > 10 else None
    for k in list(d.keys()):
        if k in ("qid","orig"): continue
        v = d.get(k)
        if not v or (isinstance(v, str) and len(v) < 20):
            d[k] = None
    return d


def main():
    print("[load]")
    val = pd.read_parquet(P / "parquet/val.parquet")
    pval = pd.read_parquet(P / "parquet/pseudo_val.parquet")
    val_r1 = json.loads((P / "expansions/val_expansions_qwen3_14b.json").read_text())
    val_mv = json.loads((P / "expansions/val_multiview_qwen3_14b.json").read_text())
    val_verify = json.loads((P / "expansions/val_verify_reasoning_qwen3_14b.json").read_text())
    pval_r1 = json.loads((P / "expansions/pseudo_val_expansions_qwen3_14b.json").read_text())
    pval_mv = json.loads((P / "expansions/pseudo_val_multiview_qwen3_14b.json").read_text())
    pval_verify_path = P / "expansions/pseudo_val_verify_reasoning_qwen3_14b.json"
    pval_verify = json.loads(pval_verify_path.read_text()) if pval_verify_path.exists() else {}

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    with open(P / "graph/court_to_bges_v1.pkl", "rb") as f:
        court_to_bges = pickle.load(f)

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

    def gather_rows(df, exp_r1, exp_mv):
        rows = []
        for r in df.itertuples():
            rows.append(build_row(r.query_id, r.query,
                                  exp_r1.get(r.query_id, {}).get("parsed"),
                                  exp_mv.get(r.query_id, {}).get("parsed")))
        return rows

    val_rows = gather_rows(val, val_r1, val_mv)
    pval_rows = gather_rows(pval, pval_r1, pval_mv)

    # Pre-warm all queries
    distinct = set()
    for rows in (val_rows, pval_rows):
        for r in rows:
            for k in MODES:
                if r.get(k): distinct.add(r[k])
    print(f"[warm] {len(distinct)} distinct query variants")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    val_golds = [split_cites(g) for g in val["gold_citations"]]
    pval_golds = [split_cites(g) for g in pval["gold_citations"]]

    def predict(r, additional, n_law, n_court, target, use_gC, use_pG, gw, bge_gw):
        rs_law, rs_court = [], []
        for k in MODES:
            v = r.get(k)
            if v: rs_law.append(laws_for(v)); rs_court.append(courts_for(v))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * gw
        if use_gC and court_to_bges:
            bge_freq = Counter()
            for d, _ in court_top[:BGE_GRAPH_POOL]:
                for bge in court_to_bges.get(d, []):
                    bge_freq[bge] += 1
            for bge, freq in bge_freq.items():
                cs[bge] = cs.get(bge, 0.0) + freq * bge_gw
            court_top = sorted(cs.items(), key=lambda x: -x[1])
        law_top = sorted(ls.items(), key=lambda x: -x[1])
        base = [d for d, _ in law_top[:n_law]] + [d for d, _ in court_top[:n_court]]
        if use_pG and additional:
            seen = set(); out = []
            for c in base + list(additional):
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target]
        return base[:target]

    print(f"\n{'='*100}")
    print(f"  {'config':<35} {'val_F1':>8} {'pval_F1':>9} {'combined':>10}")
    print(f"{'='*100}")
    rows_per_config = []
    for name, nL, nC, tgt, gC, pG, gw, bgw in CONFIGS:
        val_preds = []
        for r in val_rows:
            v = (val_verify.get(r["qid"]) or {}).get("parsed") or {}
            adds = list(v.get("additional", []) or [])
            val_preds.append(predict(r, adds, nL, nC, tgt, gC, pG, gw, bgw))
        pval_preds = []
        for r in pval_rows:
            v = (pval_verify.get(r["qid"]) or {}).get("parsed") or {}
            adds = list(v.get("additional", []) or [])
            pval_preds.append(predict(r, adds, nL, nC, tgt, gC, pG, gw, bgw))
        v_f1 = macro_f1(val_preds, val_golds)
        p_f1 = macro_f1(pval_preds, pval_golds)
        # combined macro
        all_preds = val_preds + pval_preds
        all_golds = val_golds + pval_golds
        c_f1 = macro_f1(all_preds, all_golds)
        rows_per_config.append((name, v_f1, p_f1, c_f1))
        print(f"  {name:<35} {v_f1:>8.4f} {p_f1:>9.4f} {c_f1:>10.4f}")

    print(f"\n[best by combined]")
    rows_per_config.sort(key=lambda x: -x[3])
    for name, v_f1, p_f1, c_f1 in rows_per_config[:5]:
        print(f"  {name:<35} val={v_f1:.4f}  pval={p_f1:.4f}  combined={c_f1:.4f}")


if __name__ == "__main__":
    main()
