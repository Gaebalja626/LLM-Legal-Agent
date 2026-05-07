"""
Stage 7: try harder with reranker — explore hybrid fusion with graph score.

Findings from 06_val_rerank.py:
  - Pure cross-encoder rerank ≪ baseline (0.089 vs 0.145).
  - Reranker drops graph boost signal which is doing most of the lifting.
  - Smaller K_in (50) > larger (300) — reranker hurts as candidates expand.

Hypotheses tested here:
  H1: rerank_query='kw' (German kw+codes) better than English orig.
  H2: Hybrid = z-norm(reranker) + graph_w * graph_freq, reuse Stage-2 logic.
  H3: Reranker as one of multiple rankings inside RRF (orig-search, kw-search,
      hyde-search, reranker-rank) — score-free fusion that respects graph.
"""
import os, json, pickle, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
CIT_PKL  = P / "parquet/citation_text_v2.pkl"

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W_DEFAULT = 0.05


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

def zmin_norm(scores):
    """min-max normalize so min=0, max=1; safe on constants."""
    if not scores: return {}
    arr = np.array(list(scores.values()), dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    rng = (hi - lo) or 1.0
    return {k: (v - lo) / rng for k, v in scores.items()}


def main():
    print("[load] resources")
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_PATH.read_text())
    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)

    print("[load] BGE-M3 + reranker")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256
    rerank = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")

    enc_cache, law_cache, court_cache = {}, {}, {}

    def encode_q(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]
    def laws_for(q):
        if q not in law_cache:
            v = encode_q(q)
            s, i = law_index.search(v, K_LAW_RAW)
            law_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_cache[q]
    def courts_for(q):
        if q not in court_cache:
            v = encode_q(q)
            s, i = court_index.search(v, K_COURT_RAW)
            court_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_cache[q]

    rows = []
    for vq in val.itertuples():
        exp = val_exp.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde_q = exp.get("hyde") or None
        if hyde_q and len(hyde_q) < 50: hyde_q = None
        rows.append({"qid": vq.query_id, "orig": vq.query, "kw": kw_q, "hyde": hyde_q,
                     "gold": split_cites(vq.gold_citations)})

    distinct = set()
    for r in rows:
        for k in ("orig", "kw", "hyde"):
            if r[k]: distinct.add(r[k])
    print(f"[search] pre-warming {len(distinct)} distinct queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time() - t:.1f}s")

    rerank_cache = {}  # (rq, citation) -> score

    def rerank_pairs(rq, cits, batch_size=32):
        if not cits: return {}
        # which need scoring
        todo = [c for c in cits if (rq, c) not in rerank_cache]
        if todo:
            pairs = [(rq, cit2text.get(c, c)) for c in todo]
            scores = rerank.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=False, convert_to_numpy=True)
            for c, s in zip(todo, scores.tolist()):
                rerank_cache[(rq, c)] = float(s)
        return {c: rerank_cache[(rq, c)] for c in cits}

    def stage2_scores(r, K_law_in=200, K_court_in=200):
        """Return law_scores_dict, court_scores_dict, court_ranked (sorted)."""
        qs = [r[k] for k in ("orig", "kw", "hyde") if r.get(k)]
        if not qs: qs = [r["orig"]]
        law_rankings = [laws_for(q)[:K_law_in] for q in qs]
        court_rankings = [courts_for(q)[:K_court_in] for q in qs]
        law_scores = rrf_merge(law_rankings)
        court_scores = rrf_merge(court_rankings)
        court_ranked = sorted(court_scores.items(), key=lambda x: -x[1])
        # graph boost on laws
        graph_freq = Counter()
        for d, _ in court_ranked[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                graph_freq[law] += 1
        for law, freq in graph_freq.items():
            law_scores[law] = law_scores.get(law, 0.0) + freq * GRAPH_W_DEFAULT
        return law_scores, court_scores, court_ranked

    def predict_hybrid(r, mode, rerank_query, K_in=100, n_law=15, n_court=10,
                        rerank_w=1.0, graph_w_keep=GRAPH_W_DEFAULT):
        """
        mode='hybrid_norm': zmin(rerank) + zmin(stage2_score)
        mode='rrf_with_rerank': RRF over (orig, kw, hyde, rerank_rank)
        mode='rerank_then_graph': rerank top-K, then re-add graph_freq * w
        """
        law_s, court_s, court_ranked = stage2_scores(r)
        # candidate pool from stage2
        law_pool = [d for d, _ in sorted(law_s.items(), key=lambda x: -x[1])[:K_in]]
        court_pool = [d for d, _ in court_ranked[:K_in]]

        rq = r.get(rerank_query) or r["orig"]
        rs_law = rerank_pairs(rq, law_pool)
        rs_court = rerank_pairs(rq, court_pool)

        if mode == "hybrid_norm":
            # both stage2 and reranker normalized to [0,1]
            stg_law = zmin_norm({c: law_s[c] for c in law_pool})
            stg_court = zmin_norm({c: court_s[c] for c in court_pool})
            rl = zmin_norm(rs_law); rc = zmin_norm(rs_court)
            mixed_law  = {c: stg_law.get(c, 0) + rerank_w * rl.get(c, 0) for c in law_pool}
            mixed_court = {c: stg_court.get(c, 0) + rerank_w * rc.get(c, 0) for c in court_pool}
            law_top = sorted(mixed_law.items(), key=lambda x: -x[1])[:n_law]
            court_top = sorted(mixed_court.items(), key=lambda x: -x[1])[:n_court]

        elif mode == "rrf_with_rerank":
            # build sub-rankings, fuse with RRF
            qs = [r[k] for k in ("orig","kw","hyde") if r.get(k)]
            base_law_rks = [laws_for(q)[:K_in] for q in qs]
            base_court_rks = [courts_for(q)[:K_in] for q in qs]
            rerank_law_rk = sorted(rs_law.items(), key=lambda x:-x[1])
            rerank_court_rk = sorted(rs_court.items(), key=lambda x:-x[1])
            law_fuse = rrf_merge(base_law_rks + [rerank_law_rk])
            court_fuse = rrf_merge(base_court_rks + [rerank_court_rk])
            # graph boost on fused law scores
            graph_freq = Counter()
            for d, _ in sorted(court_fuse.items(), key=lambda x:-x[1])[:GRAPH_POOL]:
                for law in court_to_laws.get(d, []):
                    graph_freq[law] += 1
            for law, freq in graph_freq.items():
                law_fuse[law] = law_fuse.get(law, 0.0) + freq * graph_w_keep
            law_top = sorted(law_fuse.items(), key=lambda x: -x[1])[:n_law]
            court_top = sorted(court_fuse.items(), key=lambda x: -x[1])[:n_court]

        elif mode == "rerank_then_graph":
            # take top-K2 by reranker, then add graph_freq*w as boost on law side only
            K2 = 60  # secondary trim
            law_re_top = sorted(rs_law.items(), key=lambda x:-x[1])[:K2]
            court_re_top = sorted(rs_court.items(), key=lambda x:-x[1])[:K2]
            law_re_dict = {c: s for c, s in law_re_top}
            court_re_dict = {c: s for c, s in court_re_top}
            # normalize
            law_re_dict = zmin_norm(law_re_dict)
            court_re_dict = zmin_norm(court_re_dict)
            graph_freq = Counter()
            for d, _ in court_re_top[:GRAPH_POOL]:
                for law in court_to_laws.get(d, []):
                    graph_freq[law] += 1
            for law, freq in graph_freq.items():
                law_re_dict[law] = law_re_dict.get(law, 0.0) + freq * graph_w_keep
            law_top = sorted(law_re_dict.items(), key=lambda x: -x[1])[:n_law]
            court_top = court_re_top[:n_court]
        else:
            raise ValueError(mode)

        return [d for d, _ in law_top] + [d for d, _ in court_top]

    golds = [r["gold"] for r in rows]

    print("\n[H1] rerank_query options (mode=hybrid_norm, K_in=100, nL=15, nC=10)")
    for rq in ("orig", "hyde", "kw"):
        preds = [predict_hybrid(r, "hybrid_norm", rq) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  rq={rq:<5}: F1={f1:.4f}")

    print("\n[H2] hybrid_norm rerank_w sweep (rq=orig, K_in=100)")
    for rw in [0.0, 0.25, 0.5, 1.0, 2.0]:
        preds = [predict_hybrid(r, "hybrid_norm", "orig", rerank_w=rw) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  rerank_w={rw:>4}: F1={f1:.4f}")

    print("\n[H3] rrf_with_rerank (rq=orig, K_in=100)")
    for rq in ("orig", "kw", "hyde"):
        preds = [predict_hybrid(r, "rrf_with_rerank", rq) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  rq={rq:<5}: F1={f1:.4f}")

    print("\n[H4] rerank_then_graph (rq=orig, K_in=100)")
    for rq in ("orig", "kw", "hyde"):
        preds = [predict_hybrid(r, "rerank_then_graph", rq) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  rq={rq:<5}: F1={f1:.4f}")

    print("\n[H5] best mode + (n_law, n_court) sweep")
    # use best mode discovered above; try a couple promising combos
    for mode in ("hybrid_norm", "rrf_with_rerank"):
        for nL, nC in [(15, 10), (20, 12), (25, 15)]:
            preds = [predict_hybrid(r, mode, "orig", n_law=nL, n_court=nC) for r in rows]
            f1 = macro_f1(preds, golds)
            print(f"  mode={mode:<18} nL={nL:>2} nC={nC:>2}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
