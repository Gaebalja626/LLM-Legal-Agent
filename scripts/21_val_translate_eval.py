"""
Stage 21: evaluate adding `q_translate_de` (natural German translation of query)
as a 4th retrieval variant alongside orig/kw/hyde.
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
TRANS_PATH = P / "expansions/val_translation_qwen3_14b.json"

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05


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
    trans = json.loads(TRANS_PATH.read_text())

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

    rows = []
    for vq in val.itertuples():
        exp = val_exp.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde_q = exp.get("hyde") or None
        if hyde_q and len(hyde_q) < 50: hyde_q = None
        translate_q = (trans.get(vq.query_id) or {}).get("translation_de") or None
        if translate_q and len(translate_q) < 30: translate_q = None
        rows.append({"qid": vq.query_id, "orig": vq.query, "kw": kw_q, "hyde": hyde_q,
                     "trans": translate_q,
                     "gold": split_cites(vq.gold_citations)})

    distinct = set()
    for r in rows:
        for k in ("orig","kw","hyde","trans"):
            if r[k]: distinct.add(r[k])
    print(f"[search] pre-warming {len(distinct)} distinct queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    # Show what the translations look like
    print("\n[trans] sample translations:")
    for r in rows[:3]:
        if r["trans"]:
            print(f"  [{r['qid']}] {r['trans'][:200]}...")

    def predict(r, modes, n_law=15, n_court=10):
        rs_law, rs_court = [], []
        for k in modes:
            if r.get(k):
                rs_law.append(laws_for(r[k]))
                rs_court.append(courts_for(r[k]))
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
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    golds = [r["gold"] for r in rows]

    print("\n[ablation A] include trans (nL=15 nC=10)")
    print(f"{'modes':<28}  F1")
    for label, modes in [
        ("orig+kw+hyde (baseline)",       ("orig","kw","hyde")),
        ("orig+kw+hyde+trans",            ("orig","kw","hyde","trans")),
        ("orig+trans",                    ("orig","trans")),
        ("kw+hyde+trans",                 ("kw","hyde","trans")),
        ("trans only",                    ("trans",)),
        ("orig+kw+trans (no hyde)",       ("orig","kw","trans")),
        ("orig+hyde+trans (no kw)",       ("orig","hyde","trans")),
    ]:
        preds = [predict(r, modes) for r in rows]
        f1 = macro_f1(preds, golds)
        per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
        print(f"  {label:<32}: {f1:.4f}  {per_q}")

    print("\n[ablation B] best mode + size sweep")
    for label, modes in [
        ("orig+kw+hyde+trans",            ("orig","kw","hyde","trans")),
    ]:
        for nL, nC in [(15,10),(20,10),(20,12),(25,15),(25,10),(30,10)]:
            preds = [predict(r, modes, n_law=nL, n_court=nC) for r in rows]
            f1 = macro_f1(preds, golds)
            print(f"  {label} nL={nL} nC={nC}: {f1:.4f}")


if __name__ == "__main__":
    main()
