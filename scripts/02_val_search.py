"""
Stage 2: multi-query dense retrieval with graph boost + ablation on val.

Builds 1-3 query variants per question (orig / kw / hyde), runs split BGE-M3
search against law/court FAISS indexes, fuses with RRF, boosts laws cited by
top retrieved courts, then writes a Macro-F1 table per ablation × output-size.

Caches every distinct query encoding to keep wall time reasonable.
"""
import os, json, pickle, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"


def split_cites(s):
    if pd.isna(s) or s == "":
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def f1_per_query(pred, gold):
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    pr = tp / len(p)
    rc = tp / len(g)
    return 2 * pr * rc / (pr + rc)


def macro_f1(preds, golds):
    return float(np.mean([f1_per_query(p, g) for p, g in zip(preds, golds)]))


def rrf_merge(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking):
            scores[doc] += 1.0 / (k + rank + 1)
    return scores


def main():
    print("[load] data + indexes")
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_expansions = json.loads(EXP_PATH.read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)

    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))

    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    print(f"  law_index={law_index.ntotal:,}, court_index={court_index.ntotal:,}, graph={len(court_to_laws):,}")

    print("[load] BGE-M3")
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.max_seq_length = 256

    enc_cache = {}
    def encode(q):
        if q not in enc_cache:
            v = model.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False).astype("float32")
            enc_cache[q] = v
        return enc_cache[q]

    K_LAW, K_COURT = 200, 200
    law_search_cache = {}
    court_search_cache = {}

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

    # Pre-build query variants for each row
    rows = []
    for vq in val.itertuples():
        exp = val_expansions.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde_q = exp.get("hyde") or None
        if hyde_q and len(hyde_q) < 50:
            hyde_q = None
        rows.append({
            "qid": vq.query_id,
            "orig": vq.query,
            "kw": kw_q,
            "hyde": hyde_q,
            "gold": split_cites(vq.gold_citations),
        })

    # Pre-warm: run all distinct queries through faiss
    distinct = set()
    for r in rows:
        for k in ("orig", "kw", "hyde"):
            if r[k]:
                distinct.add(r[k])
    print(f"[search] pre-warming {len(distinct)} distinct queries")
    t0 = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time() - t0:.1f}s")

    def predict(r, modes, out_n_law=25, out_n_court=15,
                graph_pool=100, graph_w=0.05):
        qs = [r[m] for m in modes if r.get(m)]
        if not qs:
            qs = [r["orig"]]
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

    golds = [r["gold"] for r in rows]

    print("\n[ablation] mode comparison (out_law=25, out_court=15, graph_w=0.05)")
    print(f"{'mode':<14} {'F1':>8}  per-query F1s")
    for name, modes in [
        ("orig",    ("orig",)),
        ("kw",      ("kw",)),
        ("hyde",    ("hyde",)),
        ("orig+kw", ("orig", "kw")),
        ("orig+hyde", ("orig", "hyde")),
        ("kw+hyde", ("kw", "hyde")),
        ("all",     ("orig", "kw", "hyde")),
    ]:
        preds = [predict(r, modes) for r in rows]
        f1 = macro_f1(preds, golds)
        per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
        print(f"{name:<14} {f1:>8.4f}  {per_q}")

    print("\n[ablation] graph_w sweep on best mode `all`")
    for gw in [0.0, 0.03, 0.05, 0.1, 0.2]:
        preds = [predict(r, ("orig", "kw", "hyde"), graph_w=gw) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  graph_w={gw:>4}: F1={f1:.4f}")

    print("\n[ablation] (out_n_law, out_n_court) sweep on `all` (graph_w=0.05)")
    for nL in [15, 20, 25, 30, 40]:
        for nC in [10, 15, 20, 25]:
            preds = [predict(r, ("orig","kw","hyde"), out_n_law=nL, out_n_court=nC) for r in rows]
            f1 = macro_f1(preds, golds)
            print(f"  nL={nL:>2} nC={nC:>2}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
