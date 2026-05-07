"""
Sanity: contextual augmentation + reranker on BGE pool.

Pool: same 1067 BGE pool from stage 45 (cached labels).
4 modes:
  - raw: BGE-M3 dense over raw text → top-K
  - aug: BGE-M3 dense over (context_en + context_de + raw) → top-K
  - raw + rerank: top-K_in raw → BGE-reranker-v2-m3 → top-K
  - aug + rerank: top-K_in aug → BGE-reranker-v2-m3 → top-K

For each, measure top-K hits per val query (gold BGE only).
Use multi-view 6 query variants and union the results.
"""
import os, json, pickle, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
MV_PATH  = P / "expansions/val_multiview_qwen3_14b.json"
LABELS_PATH = P / "expansions/sanity_contextual_labels.json"
CIT_PKL  = P / "parquet/citation_text_v2.pkl"

MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]


def main():
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_PATH.read_text())
    mv = json.loads(MV_PATH.read_text())
    labels = json.loads(LABELS_PATH.read_text())
    with open(P / "indexes/bge_only_doc_ids_v2.pkl", "rb") as f:
        BGE_DOC_IDS = pickle.load(f)
    BGE_VOCAB = set(BGE_DOC_IDS)
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)

    pool_keep = []
    raw_texts = []
    aug_texts = []
    for c, lab in labels.items():
        p = lab.get("parsed")
        if not p: continue
        ctx_de = p.get("context_de", "") or ""
        ctx_en = p.get("context_en", "") or ""
        raw = cit2text.get(c, "") or ""
        if not raw: continue
        pool_keep.append(c)
        raw_texts.append(raw)
        aug_texts.append(f"{ctx_en} {ctx_de} {raw}".strip())
    print(f"[pool] {len(pool_keep)} BGEs with labels")

    print("[load] BGE-M3 encoder")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256
    print("[encode] raw + aug")
    t = time.time()
    raw_emb = enc.encode(raw_texts, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64).astype("float32")
    aug_emb = enc.encode(aug_texts, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64).astype("float32")
    print(f"  encoded in {time.time()-t:.1f}s")

    raw_idx = faiss.IndexFlatIP(raw_emb.shape[1]); raw_idx.add(raw_emb)
    aug_idx = faiss.IndexFlatIP(aug_emb.shape[1]); aug_idx.add(aug_emb)

    enc_cache = {}
    def encode_q(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False).astype("float32")
        return enc_cache[q]

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
    pool_set = set(pool_keep)

    print("[load] BGE-reranker-v2-m3")
    rerank = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")

    def search_idx(idx, qv, k):
        s, i = idx.search(qv, k)
        return [pool_keep[idx_] for idx_ in i[0]]

    def rerank_texts(query, candidates, candidate_texts, k_out):
        if not candidates: return []
        pairs = [(query, t) for t in candidate_texts]
        scores = rerank.predict(pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: -x[1])
        return [c for c, _ in ranked[:k_out]]

    def evaluate(label, get_topK):
        # get_topK(r, K) returns list of citations
        for K_eval in [3, 5, 10, 20, 30]:
            total_hits = 0; total_gold = 0
            for r in rows:
                gold = [g for g in r["gold"] if g in BGE_VOCAB and g in pool_set]
                if not gold: continue
                total_gold += len(gold)
                got = get_topK(r, K_eval)
                total_hits += sum(1 for g in gold if g in got)
            print(f"  {label:<28} K={K_eval:>3}  hits {total_hits}/{total_gold} = {total_hits/max(1,total_gold):.3f}")

    # Mode 1: raw, no rerank
    def get_raw(r, K):
        union = []
        for k in MODES:
            v = r.get(k)
            if v:
                qv = encode_q(v)
                top = search_idx(raw_idx, qv, K)
                union.extend(top)
        # dedup keeping order
        seen = set(); out = []
        for c in union:
            if c not in seen:
                seen.add(c); out.append(c)
        return out[:K]

    # Mode 2: aug, no rerank
    def get_aug(r, K):
        union = []
        for k in MODES:
            v = r.get(k)
            if v:
                qv = encode_q(v)
                top = search_idx(aug_idx, qv, K)
                union.extend(top)
        seen = set(); out = []
        for c in union:
            if c not in seen:
                seen.add(c); out.append(c)
        return out[:K]

    # Mode 3 / 4: rerank top-K_in candidates → take top-K out
    def get_with_rerank(r, K, idx, K_in):
        # gather union pool
        union = []
        for k in MODES:
            v = r.get(k)
            if v:
                qv = encode_q(v)
                top = search_idx(idx, qv, K_in)
                union.extend(top)
        seen = set(); cands = []
        for c in union:
            if c not in seen:
                seen.add(c); cands.append(c)
        # Rerank with raw text snippets (more legible) — try both raw and aug texts
        cand_texts = [cit2text.get(c, "") or "" for c in cands]
        # use the strongest "query" — try orig English narrative first
        # but we'd like to also use German variants — pick orig for now
        return rerank_texts(r["orig"], cands, cand_texts, K)

    print("\n=== Recall comparison (top-K of pool=1067, gold=69 BGE) ===")
    print("\n[mode 1] raw, no rerank")
    evaluate("raw", get_raw)
    print("\n[mode 2] aug (context+text), no rerank")
    evaluate("aug", get_aug)
    print("\n[mode 3] raw + rerank (K_in=30, query=orig English)")
    evaluate("raw+rerank30", lambda r, K: get_with_rerank(r, K, raw_idx, 30))
    print("\n[mode 4] aug + rerank (K_in=30, query=orig English)")
    evaluate("aug+rerank30", lambda r, K: get_with_rerank(r, K, aug_idx, 30))
    print("\n[mode 5] aug + rerank K_in=50")
    evaluate("aug+rerank50", lambda r, K: get_with_rerank(r, K, aug_idx, 50))
    print("\n[mode 6] aug + rerank K_in=80")
    evaluate("aug+rerank80", lambda r, K: get_with_rerank(r, K, aug_idx, 80))

    # Also rerank with German narrative (hyde)
    def get_with_rerank_hyde(r, K, idx, K_in):
        union = []
        for k in MODES:
            v = r.get(k)
            if v:
                qv = encode_q(v); top = search_idx(idx, qv, K_in); union.extend(top)
        seen = set(); cands = []
        for c in union:
            if c not in seen:
                seen.add(c); cands.append(c)
        cand_texts = [cit2text.get(c, "") or "" for c in cands]
        rerank_q = r.get("hyde") or r["orig"]
        return rerank_texts(rerank_q, cands, cand_texts, K)

    print("\n[mode 7] aug + rerank K_in=50, query=German hyde")
    evaluate("aug+rerank50_hyde", lambda r, K: get_with_rerank_hyde(r, K, aug_idx, 50))


if __name__ == "__main__":
    main()
