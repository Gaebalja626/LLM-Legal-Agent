"""
Deeper EDA: with C applied (val F1 0.2193), what's still wrong?

For each val query:
  - gold split into laws / BGE-courts / docket-courts
  - prediction split into same buckets
  - TP / FP / FN per bucket
  - For FN: where does it sit in our retrieval ranking? (rank in top-200/1000/5000, or "never in pool")
  - For FP: what kind of laws/courts are we wrongly outputting?

Also count: across all 10 queries, where is the leak?
"""
import os, json, pickle, re
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
CIT_PKL  = P / "parquet/citation_text_v2.pkl"

K_LAW = 200
K_COURT = 200
GRAPH_POOL = 100
GRAPH_W = 0.05
BGE_GRAPH_POOL = 100
BGE_GRAPH_W = 0.05
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")

BGE_re = re.compile(r"^BGE\s")


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]

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
    LAW_VOCAB = set(LAW_DOC_IDS); COURT_VOCAB = set(COURT_DOC_IDS)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    with open(P / "graph/court_to_bges_v1.pkl", "rb") as f:
        court_to_bges = pickle.load(f)
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)

    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256
    enc_cache, law_cache, court_cache = {}, {}, {}
    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]
    def laws_for(q, k=K_LAW):
        v = encode(q); s, i = law_index.search(v, k)
        return [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
    def courts_for(q, k=K_COURT):
        v = encode(q); s, i = court_index.search(v, k)
        return [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]

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

    def predict_full(r, n_law=15, n_court=3):
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
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        # C: court→BGE graph
        bge_freq = Counter()
        for d, _ in court_top[:BGE_GRAPH_POOL]:
            for bge in court_to_bges.get(d, []):
                bge_freq[bge] += 1
        for bge, freq in bge_freq.items():
            cs[bge] = cs.get(bge, 0.0) + freq * BGE_GRAPH_W
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        law_top = sorted(ls.items(), key=lambda x: -x[1])
        return ([d for d, _ in law_top[:n_law]] + [d for d, _ in court_top[:n_court]],
                law_top, court_top)

    def categorize(c):
        if c in LAW_VOCAB: return "law"
        if c not in COURT_VOCAB: return "neither"
        if BGE_re.match(c): return "bge"
        return "docket"

    print("=" * 80)
    print("Per-query breakdown (config: nL=15 nC=3 with C graph)")
    print("=" * 80)
    print(f"\n{'qid':<8} {'gold(L|BGE|D)':<14} {'pred(L|BGE|D)':<14} {'TP_L':>4} {'TP_BGE':>6} {'TP_D':>4}")
    qid_data = {}
    for r in rows:
        gold = r["gold"]
        pred, law_top, court_top = predict_full(r)
        gold_cats = Counter(categorize(c) for c in gold)
        pred_cats = Counter(categorize(c) for c in pred)
        tp = set(pred) & set(gold)
        tp_cats = Counter(categorize(c) for c in tp)
        print(f"{r['qid']:<8} "
              f"{gold_cats['law']:>2}|{gold_cats['bge']:>3}|{gold_cats['docket']:>3}     "
              f"{pred_cats['law']:>2}|{pred_cats['bge']:>3}|{pred_cats['docket']:>3}     "
              f"{tp_cats['law']:>4} {tp_cats['bge']:>6} {tp_cats['docket']:>4}")
        qid_data[r["qid"]] = (gold, pred, law_top, court_top)
    print()

    # Now drill into what we're missing per category
    print("=" * 80)
    print("What we're missing (FN analysis)")
    print("=" * 80)
    miss_by_cat = Counter()
    miss_in_pool_by_cat = Counter()  # missed but at least in our top-K pool
    miss_outside_pool_by_cat = Counter()  # never even in the pool
    for r in rows:
        gold, pred, law_top, court_top = qid_data[r["qid"]]
        pred_set = set(pred)
        law_pool_top = {d for d, _ in law_top[:200]}
        court_pool_top = {d for d, _ in court_top[:200]}
        for g in gold:
            if g in pred_set: continue
            cat = categorize(g)
            miss_by_cat[cat] += 1
            in_pool = (g in law_pool_top) or (g in court_pool_top)
            if in_pool:
                miss_in_pool_by_cat[cat] += 1
            else:
                miss_outside_pool_by_cat[cat] += 1
    print(f"  category   total_miss  in_pool_top200  outside_pool")
    for cat in ("law", "bge", "docket"):
        m = miss_by_cat.get(cat, 0)
        ip = miss_in_pool_by_cat.get(cat, 0)
        op = miss_outside_pool_by_cat.get(cat, 0)
        print(f"  {cat:<10} {m:>10}  {ip:>14}  {op:>12}")

    # FP analysis: what kinds of FPs are in our prediction?
    print(f"\n{'='*80}")
    print("FP analysis (what we wrongly output)")
    print(f"{'='*80}")
    fp_by_cat = Counter()
    for r in rows:
        gold, pred, _, _ = qid_data[r["qid"]]
        gold_set = set(gold)
        for p in pred:
            if p not in gold_set:
                fp_by_cat[categorize(p)] += 1
    print(f"  total FP per category: {dict(fp_by_cat)}")

    # For each query, show the missed BGE specifically (since BGE is what graph C targets)
    print(f"\n{'='*80}")
    print("Missed gold BGEs — where are they in our court ranking?")
    print(f"{'='*80}")
    for r in rows:
        gold, pred, law_top, court_top = qid_data[r["qid"]]
        gold_bges = [g for g in gold if categorize(g) == "bge"]
        if not gold_bges: continue
        court_rank = {d: i for i, (d, _) in enumerate(court_top)}
        print(f"\n  {r['qid']}: gold BGEs ({len(gold_bges)}) — pred has BGEs: {sum(1 for p in pred if categorize(p)=='bge')}")
        for g in gold_bges[:8]:
            rank = court_rank.get(g, None)
            if rank is None:
                print(f"    {g!r}: NOT in our court pool")
            else:
                print(f"    {g!r}: rank {rank+1} in court_top")

    # For docket court that're missed — same analysis
    print(f"\n{'='*80}")
    print("Missed gold docket courts — where in our court ranking?")
    print(f"{'='*80}")
    for r in rows:
        gold, pred, law_top, court_top = qid_data[r["qid"]]
        gold_dockets = [g for g in gold if categorize(g) == "docket"]
        if not gold_dockets: continue
        court_rank = {d: i for i, (d, _) in enumerate(court_top)}
        in_pool = sum(1 for g in gold_dockets if g in court_rank)
        outside = len(gold_dockets) - in_pool
        print(f"  {r['qid']}: gold dockets ({len(gold_dockets)}) — in pool: {in_pool}  outside: {outside}")
        for g in gold_dockets[:5]:
            rank = court_rank.get(g)
            if rank is None:
                print(f"    [outside] {g!r}")
            else:
                print(f"    [rank {rank+1}] {g!r}")


if __name__ == "__main__":
    main()
