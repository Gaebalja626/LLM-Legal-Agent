"""
Stage 37: explore K_COURT (FAISS top-K for courts).

Step 1: at K_COURT ∈ {200, 500, 1000, 2000, 5000}, measure court recall ceiling
        per query (how many gold courts appear in top-K of any of our 6 query
        variants combined).
Step 2: if K↑ helps recall, run F1 with that K (final output still nC=0 — court
        only feeds graph boost).
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
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
MV_PATH  = P / "expansions/val_multiview_qwen3_14b.json"

K_LAW = 200
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

    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256
    enc_cache = {}
    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]

    def laws_for(q, k):
        v = encode(q); s, i = law_index.search(v, k)
        return [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
    def courts_for(q, k):
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
    distinct = set()
    for r in rows:
        for k in MODES:
            if r.get(k): distinct.add(r[k])
    print(f"[encode] {len(distinct)} distinct queries")
    for q in distinct:
        encode(q)

    # Step 1: court recall ceiling at various K_COURT
    print("\n[step 1] court recall ceiling (union over 6 query variants)")
    print(f"{'qid':<8} {'gold_C':>6} {'K=200':>6} {'K=500':>6} {'K=1000':>7} {'K=2000':>7} {'K=5000':>7}")
    Ks = [200, 500, 1000, 2000, 5000]
    sums = {k: [0, 0] for k in Ks}  # [hits, total_gold]
    for r in rows:
        gold_c = [g for g in r["gold"] if g in COURT_VOCAB]
        n_gold = len(gold_c); gold_set = set(gold_c)
        line = f"{r['qid']:<8} {n_gold:>6}"
        for K in Ks:
            union = set()
            for k_mode in MODES:
                v = r.get(k_mode)
                if v:
                    cs = courts_for(v, K)
                    union |= {d for d, _ in cs}
            hits = len(union & gold_set)
            line += f" {hits:>6}/{n_gold:>3}"[:-2] + f" "
            sums[K][0] += hits; sums[K][1] += n_gold
        print(line)
    print()
    for K in Ks:
        h, g = sums[K]
        print(f"  K={K}: total recall = {h}/{g} = {h/max(1,g):.3f}")

    # Step 2: F1 with bigger K_COURT (graph still uses court signal)
    print("\n[step 2] F1 with various K_COURT (final output nL=15 nC=0)")
    golds = [r["gold"] for r in rows]

    def predict(r, k_court, n_law=15, n_court=0):
        rs_law, rs_court = [], []
        for k in MODES:
            v = r.get(k)
            if v:
                rs_law.append(laws_for(v, K_LAW))
                rs_court.append(courts_for(v, k_court))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    for K in Ks:
        preds = [predict(r, K) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  K_COURT={K}: F1={f1:.4f}")

    # Step 3: increase graph_pool too (use more courts in graph)
    print("\n[step 3] (K_COURT=1000) + graph_pool sweep")
    for gp in [50, 100, 200, 500, 1000]:
        preds = []
        for r in rows:
            rs_law, rs_court = [], []
            for k in MODES:
                v = r.get(k)
                if v:
                    rs_law.append(laws_for(v, K_LAW))
                    rs_court.append(courts_for(v, 1000))
            ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
            court_top = sorted(cs.items(), key=lambda x: -x[1])
            gfc = Counter()
            for d, _ in court_top[:gp]:
                for law in court_to_laws.get(d, []):
                    gfc[law] += 1
            for law, freq in gfc.items():
                ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
            law_top = sorted(ls.items(), key=lambda x: -x[1])[:15]
            preds.append([d for d, _ in law_top])
        f1 = macro_f1(preds, golds)
        print(f"  graph_pool={gp}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
