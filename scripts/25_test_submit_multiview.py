"""
Stage 25: build test submission with best multi-view config from stage-24.

Best config (val F1 = 0.1593):
  modes = orig + kw + hyde + hyde_statute + sub_q3 + trans_concise
  K_LAW=K_COURT=200, RRF k=60
  graph_pool=100, graph_w=0.05
  out_n_law=15, out_n_court=10

Inputs:
  - parquet/{val,test}.parquet
  - expansions/{val,test}_expansions_qwen3_14b.json   (round-1)
  - expansions/{val,test}_multiview_qwen3_14b.json    (multi-view 6)
  - indexes/{law,court}_v2.faiss + doc_ids
  - graph/court_to_laws_v1.pkl

Output:
  submissions/sub_v6_multiview_*_valF1_*.csv + .meta.json
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

DEFAULT_MODES = ("orig", "kw", "hyde", "hyde_statute", "sub_q3", "trans_concise")


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


def build_row(query_id, query, exp_r1, mvp):
    kws = (exp_r1 or {}).get("keywords_de") or []
    codes = (exp_r1 or {}).get("law_codes") or []
    kw_q = " ".join(codes + kws) if (kws or codes) else None
    hyde_q = (exp_r1 or {}).get("hyde") or None
    if hyde_q and len(hyde_q) < 50: hyde_q = None
    d = {
        "qid": query_id,
        "orig": query,
        "kw": kw_q,
        "hyde": hyde_q,
        "trans_natural":  (mvp or {}).get("trans_natural") or None,
        "trans_formal":   (mvp or {}).get("trans_formal")  or None,
        "trans_concise":  (mvp or {}).get("trans_concise") or None,
        "hyde_statute":   (mvp or {}).get("hyde_statute")  or None,
        "hyde_case":      (mvp or {}).get("hyde_case")     or None,
    }
    sq = (mvp or {}).get("sub_questions") or []
    for i, q in enumerate(sq[:3], 1):
        d[f"sub_q{i}"] = q if q and len(q) > 10 else None
    for k in list(d.keys()):
        if k in ("qid", "orig"): continue
        v = d.get(k)
        if not v or (isinstance(v, str) and len(v) < 20):
            d[k] = None
    return d


def main():
    out_n_law = int(os.environ.get("OUT_N_LAW", 15))
    out_n_court = int(os.environ.get("OUT_N_COURT", 10))
    graph_w = float(os.environ.get("GRAPH_W", 0.05))
    modes_env = os.environ.get("MODES", "")
    modes = tuple(modes_env.split(",")) if modes_env else DEFAULT_MODES
    print(f"[config] modes={modes}")
    print(f"[config] nL={out_n_law} nC={out_n_court} graph_w={graph_w}")

    val = pd.read_parquet(P / "parquet/val.parquet")
    test = pd.read_parquet(P / "parquet/test.parquet")
    val_r1 = json.loads((P / "expansions/val_expansions_qwen3_14b.json").read_text())
    test_r1 = json.loads((P / "expansions/test_expansions_qwen3_14b.json").read_text())
    val_mv = json.loads((P / "expansions/val_multiview_qwen3_14b.json").read_text())
    test_mv_path = P / "expansions/test_multiview_qwen3_14b.json"
    if not test_mv_path.exists():
        raise SystemExit(f"missing {test_mv_path} — run stage 22 test first")
    test_mv = json.loads(test_mv_path.read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)

    print("[load] BGE-M3")
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
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * graph_w
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:out_n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:out_n_court]]

    # val sanity
    val_rows = [build_row(vq.query_id, vq.query,
                          val_r1.get(vq.query_id, {}).get("parsed"),
                          val_mv.get(vq.query_id, {}).get("parsed")) for vq in val.itertuples()]
    distinct = set()
    for r in val_rows:
        for k in modes:
            if r.get(k): distinct.add(r[k])
    print(f"[warm] {len(distinct)} val variants")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")
    val_preds = [predict(r) for r in val_rows]
    val_golds = [split_cites(g) for g in val["gold_citations"]]
    val_f1 = macro_f1(val_preds, val_golds)
    val_per_q = {qid: f1_per_query(p, g) for qid, p, g in zip(val["query_id"], val_preds, val_golds)}
    print(f"\n[val] Macro F1 = {val_f1:.4f}")
    for qid in val["query_id"].tolist():
        print(f"  {qid}: {val_per_q[qid]:.4f}")

    # test
    test_rows = [build_row(tq.query_id, tq.query,
                           test_r1.get(tq.query_id, {}).get("parsed"),
                           test_mv.get(tq.query_id, {}).get("parsed")) for tq in test.itertuples()]
    distinct = set()
    for r in test_rows:
        for k in modes:
            if r.get(k): distinct.add(r[k])
    distinct -= set(law_cache.keys())
    print(f"\n[warm] {len(distinct)} test variants")
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
    name = f"sub_v6_multiview_oL{out_n_law}_oC{out_n_court}_gw{graph_w}_valF1_{f1_str}_{ts}"
    csv_path = SUB_DIR / f"{name}.csv"
    meta_path = SUB_DIR / f"{name}.meta.json"
    sub_df.to_csv(csv_path, index=False)
    meta = {
        "submission": csv_path.name,
        "pipeline": "Qwen3-14B-AWQ round-1 + multi-view (statute/sub_q3/concise) → BGE-M3 split dense → RRF → court→law graph boost",
        "config": {
            "llm_model": "Qwen/Qwen3-14B-AWQ",
            "embedding_model": "BAAI/bge-m3",
            "max_seq_length": 256,
            "K_LAW": K_LAW_RAW,
            "K_COURT": K_COURT_RAW,
            "graph_pool": GRAPH_POOL,
            "graph_w": graph_w,
            "out_n_law": out_n_law,
            "out_n_court": out_n_court,
            "modes": list(modes),
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
