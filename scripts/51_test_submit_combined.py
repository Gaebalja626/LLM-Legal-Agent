"""
Combined: multi-view + court→BGE graph C + Plan G additional citations.

Pipeline:
  1. multi-view retrieval (6 modes) → law/court rankings + court→law graph boost
  2. court→BGE graph 1-hop boost (NEW: surfaces BGE inside court_top)
  3. base = top-N_LAW law + top-N_COURT court
  4. + LLM verify additional citations (Plan G)
  5. dedup → final top-TARGET_SIZE

Env vars:
  N_LAW_BASE   default 15
  N_COURT_BASE default 3
  TARGET_SIZE  default 20
  USE_GRAPH_C  default 1
  USE_PLAN_G   default 1
  GRAPH_W      default 0.05
  BGE_GRAPH_W  default 0.05
"""
import os, json, pickle, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()
SUB_DIR = P / "submissions"
SUB_DIR.mkdir(exist_ok=True)

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
BGE_GRAPH_POOL = 100
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
    n_law_base = int(os.environ.get("N_LAW_BASE", 15))
    n_court_base = int(os.environ.get("N_COURT_BASE", 3))
    target_size = int(os.environ.get("TARGET_SIZE", 20))
    use_graph_c = bool(int(os.environ.get("USE_GRAPH_C", 1)))
    use_plan_g = bool(int(os.environ.get("USE_PLAN_G", 1)))
    graph_w = float(os.environ.get("GRAPH_W", 0.05))
    bge_graph_w = float(os.environ.get("BGE_GRAPH_W", 0.05))
    print(f"[config] nL_base={n_law_base} nC_base={n_court_base} target={target_size}")
    print(f"         use_graph_c={use_graph_c} use_plan_g={use_plan_g}")
    print(f"         graph_w={graph_w} bge_graph_w={bge_graph_w}")

    val = pd.read_parquet(P / "parquet/val.parquet")
    test = pd.read_parquet(P / "parquet/test.parquet")
    val_r1 = json.loads((P / "expansions/val_expansions_qwen3_14b.json").read_text())
    test_r1 = json.loads((P / "expansions/test_expansions_qwen3_14b.json").read_text())
    val_mv = json.loads((P / "expansions/val_multiview_qwen3_14b.json").read_text())
    test_mv = json.loads((P / "expansions/test_multiview_qwen3_14b.json").read_text())
    val_verify = json.loads((P / "expansions/val_verify_reasoning_qwen3_14b.json").read_text())
    test_verify = json.loads((P / "expansions/test_verify_reasoning_qwen3_14b.json").read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    if use_graph_c:
        with open(P / "graph/court_to_bges_v1.pkl", "rb") as f:
            court_to_bges = pickle.load(f)
    else:
        court_to_bges = {}

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

    def predict(r, additional=None):
        rs_law, rs_court = [], []
        for k in MODES:
            v = r.get(k)
            if v:
                rs_law.append(laws_for(v)); rs_court.append(courts_for(v))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        # Court→law graph
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * graph_w
        # Court→BGE graph (C)
        if use_graph_c and court_to_bges:
            bge_freq = Counter()
            for d, _ in court_top[:BGE_GRAPH_POOL]:
                for bge in court_to_bges.get(d, []):
                    bge_freq[bge] += 1
            for bge, freq in bge_freq.items():
                cs[bge] = cs.get(bge, 0.0) + freq * bge_graph_w
            court_top = sorted(cs.items(), key=lambda x: -x[1])
        law_top = sorted(ls.items(), key=lambda x: -x[1])
        base = [d for d, _ in law_top[:n_law_base]] + [d for d, _ in court_top[:n_court_base]]
        if additional and use_plan_g:
            seen = set(); out = []
            for c in base + list(additional):
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]
        return base[:target_size]

    # Val sanity
    val_rows = [build_row(vq.query_id, vq.query,
                          val_r1.get(vq.query_id, {}).get("parsed"),
                          val_mv.get(vq.query_id, {}).get("parsed")) for vq in val.itertuples()]
    distinct = set()
    for r in val_rows:
        for k in MODES:
            if r.get(k): distinct.add(r[k])
    for q in distinct:
        laws_for(q); courts_for(q)
    val_preds = []
    for r in val_rows:
        v = (val_verify.get(r["qid"]) or {}).get("parsed") or {}
        adds = list(v.get("additional", []) or [])
        val_preds.append(predict(r, additional=adds))
    val_golds = [split_cites(g) for g in val["gold_citations"]]
    val_f1 = macro_f1(val_preds, val_golds)
    val_per_q = {qid: f1_per_query(p, g) for qid, p, g in zip(val["query_id"], val_preds, val_golds)}
    print(f"\n[val] Macro F1 = {val_f1:.4f}")
    for qid in val["query_id"].tolist():
        print(f"  {qid}: {val_per_q[qid]:.4f}")

    # Test
    test_rows = [build_row(tq.query_id, tq.query,
                           test_r1.get(tq.query_id, {}).get("parsed"),
                           test_mv.get(tq.query_id, {}).get("parsed")) for tq in test.itertuples()]
    distinct = set()
    for r in test_rows:
        for k in MODES:
            if r.get(k): distinct.add(r[k])
    distinct -= set(law_cache.keys())
    for q in distinct:
        laws_for(q); courts_for(q)
    test_preds = []
    for r in test_rows:
        v = (test_verify.get(r["qid"]) or {}).get("parsed") or {}
        adds = list(v.get("additional", []) or [])
        test_preds.append(predict(r, additional=adds))

    rows = [{"query_id": qid, "predicted_citations": ";".join(pred)}
            for qid, pred in zip(test["query_id"], test_preds)]
    sub_df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    f1_str = f"{val_f1:.4f}".replace(".", "_")
    parts = [f"oL{n_law_base}", f"oC{n_court_base}", f"t{target_size}", f"gw{graph_w}"]
    if use_graph_c: parts.append("gC")
    if use_plan_g: parts.append("pG")
    name = f"sub_v8_combined_" + "_".join(parts) + f"_valF1_{f1_str}_{ts}"
    csv_path = SUB_DIR / f"{name}.csv"
    meta_path = SUB_DIR / f"{name}.meta.json"
    sub_df.to_csv(csv_path, index=False)
    meta = {
        "submission": csv_path.name,
        "pipeline": "multi-view + court→law graph + court→BGE graph + Plan-G additional",
        "config": {
            "modes": list(MODES),
            "n_law_base": n_law_base, "n_court_base": n_court_base,
            "graph_w": graph_w, "bge_graph_w": bge_graph_w,
            "graph_pool": GRAPH_POOL, "bge_graph_pool": BGE_GRAPH_POOL,
            "use_graph_c": use_graph_c, "use_plan_g": use_plan_g,
            "target_size": target_size,
            "K_LAW": K_LAW_RAW, "K_COURT": K_COURT_RAW,
        },
        "val_macro_f1": val_f1,
        "val_per_query_f1": val_per_q,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
