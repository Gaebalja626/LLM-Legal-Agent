"""
Stage 15: build the final test submission with round-1 + round-2 expansions.

Pipeline:
  - Round-1 expansion (codes + keywords + hyde) AND
  - Round-2 additional codes/keywords (refined_hyde optional via env USE_HYDE2)
  - dedup keywords
  - Multi-query dense (orig, kw, hyde, [hyde2]) → RRF
  - court→law graph 1-hop boost
  - top-N_law + top-N_court output

Env vars (with defaults from val ablation in 13/14):
  OUT_N_LAW=15, OUT_N_COURT=10, GRAPH_W=0.05, USE_HYDE2=0
"""
import os, json, pickle, time, sys
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


def build_row(qid, query, exp_r1, exp_r2):
    kws = list((exp_r1 or {}).get("keywords_de") or [])
    codes = list((exp_r1 or {}).get("law_codes") or [])
    hyde = (exp_r1 or {}).get("hyde") or ""
    if exp_r2:
        for c in (exp_r2.get("additional_law_codes") or []):
            if c not in codes: codes.append(c)
        for k in (exp_r2.get("additional_keywords_de") or []):
            if k not in kws: kws.append(k)
    # dedup keywords (preserve order)
    seen = set(); kws_d = []
    for k in kws:
        if k not in seen:
            seen.add(k); kws_d.append(k)
    kws = kws_d
    kw_q = " ".join(codes + kws) if (kws or codes) else None
    if hyde and len(hyde) < 50: hyde = ""
    hyde2 = (exp_r2 or {}).get("refined_hyde") or ""
    if hyde2 and len(hyde2) < 50: hyde2 = ""
    return {
        "qid": qid,
        "orig": query,
        "kw": kw_q,
        "hyde": hyde or None,
        "hyde2": hyde2 or None,
    }


def main():
    out_n_law = int(os.environ.get("OUT_N_LAW", 15))
    out_n_court = int(os.environ.get("OUT_N_COURT", 10))
    graph_w = float(os.environ.get("GRAPH_W", 0.05))
    use_hyde2 = bool(int(os.environ.get("USE_HYDE2", 0)))
    print(f"[config] nL={out_n_law} nC={out_n_court} graph_w={graph_w} use_hyde2={use_hyde2}")

    val = pd.read_parquet(P / "parquet/val.parquet")
    test = pd.read_parquet(P / "parquet/test.parquet")
    val_r1 = json.loads((P / "expansions/val_expansions_qwen3_14b.json").read_text())
    val_r2 = json.loads((P / "expansions/val_round2_qwen3_14b.json").read_text())
    test_r1 = json.loads((P / "expansions/test_expansions_qwen3_14b.json").read_text())
    test_r2_path = P / "expansions/test_round2_qwen3_14b.json"
    if not test_r2_path.exists():
        print(f"ERROR: missing {test_r2_path}")
        sys.exit(1)
    test_r2 = json.loads(test_r2_path.read_text())

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

    def predict(r):
        rs_law, rs_court = [], []
        for k in ("orig","kw","hyde"):
            if r.get(k):
                rs_law.append(laws_for(r[k]))
                rs_court.append(courts_for(r[k]))
        if use_hyde2 and r.get("hyde2"):
            rs_law.append(laws_for(r["hyde2"]))
            rs_court.append(courts_for(r["hyde2"]))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * graph_w
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:out_n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:out_n_court]]

    # Val sanity
    val_rows = []
    val_golds = []
    for vq in val.itertuples():
        r1 = val_r1.get(vq.query_id, {}).get("parsed") or {}
        r2 = val_r2.get(vq.query_id, {}).get("parsed") or {}
        val_rows.append(build_row(vq.query_id, vq.query, r1, r2))
        val_golds.append(split_cites(vq.gold_citations))

    # Pre-warm
    distinct = set()
    for r in val_rows:
        for k in ("orig","kw","hyde","hyde2"):
            if r.get(k): distinct.add(r[k])
    print(f"[warm] {len(distinct)} val queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    val_preds = [predict(r) for r in val_rows]
    val_f1 = macro_f1(val_preds, val_golds)
    val_per_q = {qid: f1_per_query(p, g)
                 for qid, p, g in zip(val["query_id"], val_preds, val_golds)}
    print(f"\n[val] Macro F1 = {val_f1:.4f}")
    for qid in val["query_id"].tolist():
        print(f"  {qid}: {val_per_q[qid]:.4f}")

    # Test
    print(f"\n[test] {len(test)} queries")
    test_rows = []
    for tq in test.itertuples():
        r1 = test_r1.get(tq.query_id, {}).get("parsed") or {}
        r2 = test_r2.get(tq.query_id, {}).get("parsed") or {}
        test_rows.append(build_row(tq.query_id, tq.query, r1, r2))
    distinct = set()
    for r in test_rows:
        for k in ("orig","kw","hyde","hyde2"):
            if r.get(k): distinct.add(r[k])
    distinct -= set(law_cache.keys())
    print(f"[warm] {len(distinct)} test queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    test_preds = [predict(r) for r in test_rows]

    rows = [{"query_id": qid, "predicted_citations": ";".join(pred)}
            for qid, pred in zip(test["query_id"], test_preds)]
    sub_df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    f1_str = f"{val_f1:.4f}".replace(".", "_")
    suffix = "with_hyde2" if use_hyde2 else "no_hyde2"
    name = f"sub_v5_round2_{suffix}_oL{out_n_law}_oC{out_n_court}_gw{graph_w}_valF1_{f1_str}_{ts}"
    csv_path = SUB_DIR / f"{name}.csv"
    meta_path = SUB_DIR / f"{name}.meta.json"
    sub_df.to_csv(csv_path, index=False)

    meta = {
        "submission": csv_path.name,
        "pipeline": "Qwen3-14B-AWQ round-1 + round-2 (iterative agentic) → BGE-M3 split dense → RRF → court→law graph boost",
        "config": {
            "llm_model": "Qwen/Qwen3-14B-AWQ",
            "embedding_model": "BAAI/bge-m3",
            "max_seq_length": 256,
            "K_LAW": K_LAW_RAW,
            "K_COURT": K_COURT_RAW,
            "out_n_law": out_n_law,
            "out_n_court": out_n_court,
            "graph_pool": GRAPH_POOL,
            "graph_w": graph_w,
            "use_hyde2": use_hyde2,
        },
        "val_macro_f1": val_f1,
        "val_per_query_f1": val_per_q,
        "n_test_predictions": len(test_preds),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
