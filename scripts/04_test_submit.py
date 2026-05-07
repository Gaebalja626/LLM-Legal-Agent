"""
Stage 4: build test submission CSV using val-best config.

Pipeline (per query):
  - LLM expansion (cached) → 3 query variants: orig + kw + hyde
  - BGE-M3 dense retrieval, split law/court FAISS, RRF fuse
  - Graph 1-hop boost on law candidates from top-pool of court hits
  - Output: top-N_law laws + top-N_court courts → semicolon-joined CSV

Writes ~/legal-ir/submissions/sub_v4_*.csv plus a meta.json with config + val F1.
Re-runs val with same config first as a sanity check (printed F1).
"""
import os, json, pickle, time, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

P = Path("~/legal-ir").expanduser()
SUB_DIR = P / "submissions"
SUB_DIR.mkdir(exist_ok=True)


def split_cites(s):
    if pd.isna(s) or s == "":
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def f1_per_query(pred, gold):
    p, g = set(pred), set(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    tp = len(p & g)
    if tp == 0: return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def macro_f1(preds, golds):
    return float(np.mean([f1_per_query(p, g) for p, g in zip(preds, golds)]))


def rrf_merge(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking):
            scores[doc] += 1.0 / (k + rank + 1)
    return scores


def build_predictor(model, law_index, court_index, LAW_DOC_IDS, COURT_DOC_IDS,
                    court_to_laws, K_LAW=200, K_COURT=200):
    enc_cache = {}
    law_search_cache = {}
    court_search_cache = {}

    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = model.encode([q], normalize_embeddings=True,
                                        convert_to_numpy=True,
                                        show_progress_bar=False).astype("float32")
        return enc_cache[q]

    def laws_for(q):
        if q not in law_search_cache:
            v = encode(q)
            s, i = law_index.search(v, K_LAW)
            law_search_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_search_cache[q]

    def courts_for(q):
        if q not in court_search_cache:
            v = encode(q)
            s, i = court_index.search(v, K_COURT)
            court_search_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_search_cache[q]

    def predict(qids_orig_kw_hyde, out_n_law=15, out_n_court=10,
                graph_pool=100, graph_w=0.05):
        # qids_orig_kw_hyde: dict with keys orig (str), kw (str|None), hyde (str|None)
        qs = [qids_orig_kw_hyde[k] for k in ("orig", "kw", "hyde") if qids_orig_kw_hyde.get(k)]
        if not qs:
            qs = [qids_orig_kw_hyde["orig"]]
        law_rankings = [laws_for(q) for q in qs]
        court_rankings = [courts_for(q) for q in qs]
        law_scores = rrf_merge(law_rankings)
        court_scores = rrf_merge(court_rankings)
        court_top = sorted(court_scores.items(), key=lambda x: -x[1])
        if graph_w > 0:
            graph_freq = Counter()
            for d, _ in court_top[:graph_pool]:
                for law in court_to_laws.get(d, []):
                    graph_freq[law] += 1
            for law, freq in graph_freq.items():
                law_scores[law] = law_scores.get(law, 0.0) + freq * graph_w
        law_top = sorted(law_scores.items(), key=lambda x: -x[1])[:out_n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:out_n_court]]

    return predict


def make_query_dict(row, expansion):
    exp = expansion.get(row["query_id"], {}).get("parsed") or {}
    kws = exp.get("keywords_de") or []
    codes = exp.get("law_codes") or []
    kw_q = " ".join(codes + kws) if (kws or codes) else None
    hyde_q = exp.get("hyde") or None
    if hyde_q and len(hyde_q) < 50:
        hyde_q = None
    return {"orig": row["query"], "kw": kw_q, "hyde": hyde_q}


def main():
    out_n_law = int(os.environ.get("OUT_N_LAW", 15))
    out_n_court = int(os.environ.get("OUT_N_COURT", 10))
    graph_w = float(os.environ.get("GRAPH_W", 0.05))
    graph_pool = int(os.environ.get("GRAPH_POOL", 100))

    print(f"[config] nL={out_n_law} nC={out_n_court} graph_w={graph_w} graph_pool={graph_pool}")
    print("[load] data + indexes")
    val = pd.read_parquet(P / "parquet/val.parquet")
    test = pd.read_parquet(P / "parquet/test.parquet")
    val_exp = json.loads((P / "expansions/val_expansions_qwen3_14b.json").read_text())
    test_exp_path = P / "expansions/test_expansions_qwen3_14b.json"
    if not test_exp_path.exists():
        print(f"ERROR: {test_exp_path} not found. Run scripts/03_test_expansion.py first.")
        sys.exit(1)
    test_exp = json.loads(test_exp_path.read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    print(f"  law={law_index.ntotal:,} court={court_index.ntotal:,} graph={len(court_to_laws):,}")

    print("[load] BGE-M3")
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.max_seq_length = 256
    predict = build_predictor(model, law_index, court_index, LAW_DOC_IDS, COURT_DOC_IDS, court_to_laws)

    # Val sanity
    val_qs = [make_query_dict(r._asdict() if hasattr(r, "_asdict") else r.to_dict(), val_exp)
              for _, r in val.iterrows()]
    val_preds = [predict(q, out_n_law=out_n_law, out_n_court=out_n_court,
                         graph_pool=graph_pool, graph_w=graph_w) for q in val_qs]
    val_golds = [split_cites(g) for g in val["gold_citations"]]
    val_f1 = macro_f1(val_preds, val_golds)
    val_per_q = {qid: f1_per_query(p, g) for qid, p, g in
                 zip(val["query_id"], val_preds, val_golds)}
    print(f"[val] Macro F1 = {val_f1:.4f}")
    for qid, f in val_per_q.items():
        print(f"  {qid}: {f:.4f}")

    # Test submission
    print(f"\n[test] generating predictions for {len(test)} queries")
    test_qs = [make_query_dict(r.to_dict(), test_exp) for _, r in test.iterrows()]
    test_preds = [predict(q, out_n_law=out_n_law, out_n_court=out_n_court,
                          graph_pool=graph_pool, graph_w=graph_w) for q in test_qs]

    rows = []
    for qid, pred in zip(test["query_id"], test_preds):
        rows.append({"query_id": qid, "predicted_citations": ";".join(pred)})
    sub_df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    f1_str = f"{val_f1:.4f}".replace(".", "_")
    name = f"sub_v4_agentic_qwen14b_oL{out_n_law}_oC{out_n_court}_gw{graph_w}_valF1_{f1_str}_{ts}"
    csv_path = SUB_DIR / f"{name}.csv"
    meta_path = SUB_DIR / f"{name}.meta.json"

    sub_df.to_csv(csv_path, index=False)
    meta = {
        "submission": csv_path.name,
        "pipeline": "Qwen3-14B-AWQ expansion (orig+kw+hyde) → BGE-M3 split dense → RRF → Graph 1-hop boost → quota output",
        "config": {
            "llm_model": "Qwen/Qwen3-14B-AWQ",
            "embedding_model": "BAAI/bge-m3",
            "max_seq_length": 256,
            "K_LAW": 200,
            "K_COURT": 200,
            "out_n_law": out_n_law,
            "out_n_court": out_n_court,
            "graph_pool": graph_pool,
            "graph_w": graph_w,
            "modes": ["orig", "kw", "hyde"],
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
