"""
Stage 12: try cheap fixes that target the failure modes seen in 11_diagnose:

  fix1: dedup keywords (val_003 had 'Vorwurf der Fluchtgefahr' 8 times)
  fix2: union default Swiss law codes into law_codes (so kw query
        always covers the dominant code abbreviations even when the
        LLM forgot, e.g. BGG/StBOG/SchKG/ZPO/BV)
  fix3: per-code search variants — for each law_code, build
        '<code> <keywords>' as its own query, RRF fuse across codes
        (gives each code its own retrieval slot)

Compare against the original baseline (stored in 02_val_search.py).
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

K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05

# Most common Swiss law abbreviations seen in train/val gold + handbook
DEFAULT_CODES = [
    "ZGB", "OR", "ZPO", "StPO", "StGB", "BGG", "BV", "IPRG", "DBG",
    "SchKG", "AHVG", "IVG", "UVG", "BVG", "StBOG", "MWSTG",
    "FZA", "AsylG", "AIG", "EMRK", "VStG", "BankG", "FINIG", "FINMAG",
    "ATSG", "KVG", "BVV2", "VAG", "USG", "RPG"
]


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
    print("[load]")
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
            v = encode(q)
            s, i = law_index.search(v, K_LAW_RAW)
            law_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_cache[q]
    def courts_for(q):
        if q not in court_cache:
            v = encode(q)
            s, i = court_index.search(v, K_COURT_RAW)
            court_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_cache[q]

    def build_row(vq, fix):
        exp = val_exp.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        hyde = exp.get("hyde") or ""
        if len(hyde) < 50: hyde = ""

        # fix1: keyword dedup (preserve first-seen order)
        seen = set(); kws_d = []
        for k in kws:
            if k not in seen:
                seen.add(k); kws_d.append(k)
        kws = kws_d

        # fix2: union default Swiss codes
        if "default_codes" in fix:
            codes_set = set(codes)
            for c in DEFAULT_CODES:
                if c not in codes_set:
                    codes.append(c)

        # base queries
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        return {
            "qid": vq.query_id,
            "orig": vq.query,
            "kw": kw_q,
            "hyde": hyde or None,
            "codes": codes,
            "keywords": kws,
            "gold": split_cites(vq.gold_citations),
        }

    def predict(r, fix=set(), n_law=15, n_court=10):
        # build query variants
        rankings_law = []
        rankings_court = []
        for k in ("orig", "kw", "hyde"):
            if r.get(k):
                rankings_law.append(laws_for(r[k]))
                rankings_court.append(courts_for(r[k]))

        if "per_code" in fix:
            # one mini-query per code: '<code> <kw1> <kw2> ...'
            kw_str = " ".join((r.get("keywords") or [])[:8])
            for c in r.get("codes") or []:
                cq = f"{c} {kw_str}".strip()
                if cq:
                    rankings_law.append(laws_for(cq))

        law_scores = rrf_merge(rankings_law)
        court_scores = rrf_merge(rankings_court)
        court_top = sorted(court_scores.items(), key=lambda x: -x[1])
        # graph boost
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            law_scores[law] = law_scores.get(law, 0.0) + freq * GRAPH_W

        law_top = sorted(law_scores.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    # baseline-ish
    print("\n[baseline] no fix (matches earlier 0.1447)")
    rows = [build_row(vq, set()) for vq in val.itertuples()]
    distinct = set()
    for r in rows:
        for k in ("orig","kw","hyde"):
            if r[k]: distinct.add(r[k])
    print(f"  pre-warm {len(distinct)} queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  warm done in {time.time()-t:.1f}s")
    golds = [r["gold"] for r in rows]
    preds = [predict(r, fix=set()) for r in rows]
    print(f"  F1 = {macro_f1(preds, golds):.4f}")

    print("\n[fix1] dedup only (already applied to all rows)")
    # rows already deduped — same as baseline; print explicit
    preds = [predict(r, fix=set()) for r in rows]
    print(f"  F1 = {macro_f1(preds, golds):.4f}  (dedup is built into row builder)")

    for label, fix in [
        ("fix2 default_codes", {"default_codes"}),
        ("fix3 per_code",      {"per_code"}),
        ("fix2+fix3",          {"default_codes", "per_code"}),
    ]:
        rows_x = [build_row(vq, fix) for vq in val.itertuples()]
        # warm any new queries
        new_qs = set()
        for r in rows_x:
            for k in ("orig","kw","hyde"):
                if r.get(k): new_qs.add(r[k])
            kw_str = " ".join((r.get("keywords") or [])[:8])
            for c in r.get("codes") or []:
                q = f"{c} {kw_str}".strip()
                if q: new_qs.add(q)
        new_qs -= set(law_cache.keys())
        if new_qs:
            t = time.time()
            for q in new_qs:
                laws_for(q); courts_for(q)
            print(f"\n  pre-warm {len(new_qs)} new queries in {time.time()-t:.1f}s")
        preds = [predict(r, fix=fix) for r in rows_x]
        f1 = macro_f1(preds, golds)
        per_q = {qid: round(f1_per_query(p, g), 3) for qid, p, g in zip(val["query_id"], preds, golds)}
        print(f"\n[{label}] F1 = {f1:.4f}")
        for qid in ["val_001","val_003","val_007","val_010"]:
            print(f"   {qid}: {per_q[qid]}")

    # combined best with output-size sweep
    print("\n[best fix combo] sweep (n_law,n_court)")
    rows_x = [build_row(vq, {"default_codes","per_code"}) for vq in val.itertuples()]
    new_qs = set()
    for r in rows_x:
        kw_str = " ".join((r.get("keywords") or [])[:8])
        for c in r.get("codes") or []:
            q = f"{c} {kw_str}".strip()
            if q: new_qs.add(q)
    new_qs -= set(law_cache.keys())
    if new_qs:
        for q in new_qs:
            laws_for(q); courts_for(q)
    for nL, nC in [(15,10),(20,12),(25,15),(20,10),(25,10),(30,10)]:
        preds = [predict(r, fix={"default_codes","per_code"}, n_law=nL, n_court=nC) for r in rows_x]
        f1 = macro_f1(preds, golds)
        print(f"  nL={nL:>2} nC={nC:>2}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
