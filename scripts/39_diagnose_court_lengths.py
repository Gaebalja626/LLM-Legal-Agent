"""
Diagnose court retrieval failures by length.

For all val gold courts:
  - their text lengths
  - which ones are in our top-K (K=200, 1000, 5000)
  - what's the length distribution of OUR top-K courts vs gold

Hypothesis: many gold courts are substantive (1000+ chars), but our top-K is
dominated by short boilerplate (<200 chars), so we can't match them.
"""
import os, json, pickle
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
GRAPH_POOL = 100
GRAPH_W = 0.05
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")


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
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)

    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256
    enc_cache = {}
    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]
    def laws_for(q, k=K_LAW):
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

    print("=== Part 1: gold court length distribution ===")
    all_gold_courts = []
    for r in rows:
        for g in r["gold"]:
            if g in COURT_VOCAB:
                all_gold_courts.append(g)
    print(f"total gold courts (val 10 queries): {len(all_gold_courts)}")
    lens = [len(cit2text.get(c, "") or "") for c in all_gold_courts]
    print(f"length stats:")
    print(f"  mean={np.mean(lens):.0f}, median={np.median(lens):.0f}, std={np.std(lens):.0f}")
    print(f"  p10={np.percentile(lens,10):.0f}  p25={np.percentile(lens,25):.0f}  p50={np.percentile(lens,50):.0f}  p75={np.percentile(lens,75):.0f}  p90={np.percentile(lens,90):.0f}")
    print(f"  min={min(lens)}  max={max(lens)}")
    bucket_counts = Counter()
    for L in lens:
        if L < 100: bucket_counts["<100"] += 1
        elif L < 200: bucket_counts["100-200"] += 1
        elif L < 500: bucket_counts["200-500"] += 1
        elif L < 1000: bucket_counts["500-1000"] += 1
        elif L < 2000: bucket_counts["1000-2000"] += 1
        else: bucket_counts["2000+"] += 1
    for b in ["<100","100-200","200-500","500-1000","1000-2000","2000+"]:
        print(f"  {b}: {bucket_counts.get(b, 0)} ({bucket_counts.get(b,0)/max(1,len(lens))*100:.0f}%)")

    print("\n=== Part 2: our retrieved top-K court length distribution ===")
    for r in rows[:3]:  # show first 3 queries' top-15 court lengths
        print(f"\n  {r['qid']}: top-15 court lengths (using all 6 query variants RRF):")
        rs_court = []
        for k in MODES:
            v = r.get(k)
            if v: rs_court.append(courts_for(v, K_LAW))
        cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])[:15]
        for c, _ in court_top:
            L = len(cit2text.get(c, "") or "")
            txt_preview = (cit2text.get(c, "") or "")[:80].replace("\n", " ")
            in_gold = "✓" if c in r["gold"] else " "
            print(f"    [{in_gold}] len={L:>5}  {c:<35}  {txt_preview}")

    print("\n=== Part 3: aggregate top-K court length stats (all 10 queries) ===")
    top_lens = []
    for r in rows:
        rs_court = []
        for k in MODES:
            v = r.get(k)
            if v: rs_court.append(courts_for(v, K_LAW))
        cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])[:30]
        for c, _ in court_top:
            top_lens.append(len(cit2text.get(c, "") or ""))
    print(f"top-30 retrieved court (across 10 queries, n={len(top_lens)}):")
    print(f"  mean={np.mean(top_lens):.0f}, median={np.median(top_lens):.0f}")
    bc = Counter()
    for L in top_lens:
        if L < 100: bc["<100"] += 1
        elif L < 200: bc["100-200"] += 1
        elif L < 500: bc["200-500"] += 1
        elif L < 1000: bc["500-1000"] += 1
        elif L < 2000: bc["1000-2000"] += 1
        else: bc["2000+"] += 1
    for b in ["<100","100-200","200-500","500-1000","1000-2000","2000+"]:
        print(f"  {b}: {bc.get(b, 0)} ({bc.get(b,0)/max(1,len(top_lens))*100:.0f}%)")

    print("\n=== Part 4: court ceiling at K=5000 — gold court length analysis ===")
    print("How many gold courts at each length bucket are recoverable in top-5000?")
    bucket_total = Counter(); bucket_hit = Counter()
    for r in rows:
        gold_c = [g for g in r["gold"] if g in COURT_VOCAB]
        union = set()
        for k in MODES:
            v = r.get(k)
            if v:
                cs = courts_for(v, 5000)
                union |= {d for d, _ in cs}
        for c in gold_c:
            L = len(cit2text.get(c, "") or "")
            if L < 100: b = "<100"
            elif L < 200: b = "100-200"
            elif L < 500: b = "200-500"
            elif L < 1000: b = "500-1000"
            elif L < 2000: b = "1000-2000"
            else: b = "2000+"
            bucket_total[b] += 1
            if c in union: bucket_hit[b] += 1
    print(f"  bucket          gold      hit_in_top5000   recall")
    for b in ["<100","100-200","200-500","500-1000","1000-2000","2000+"]:
        t = bucket_total.get(b, 0); h = bucket_hit.get(b, 0)
        if t > 0:
            print(f"  {b:<12} {t:>6}        {h:>6}            {h/t:.2f}")

    print("\n=== Part 5: substantive-only ceiling sketch ===")
    print("If we restrict court vocabulary to len>=200 ('substantive only'):")
    short_n = sum(1 for c in COURT_DOC_IDS if len(cit2text.get(c, "") or "") < 200)
    sub_n = len(COURT_DOC_IDS) - short_n
    print(f"  drop short (<200): {short_n:,} / {len(COURT_DOC_IDS):,} ({short_n/len(COURT_DOC_IDS)*100:.1f}%)")
    print(f"  remaining substantive: {sub_n:,}")
    sub_gold_total = sum(1 for r in rows for g in r["gold"]
                          if g in COURT_VOCAB and len(cit2text.get(g, "") or "") >= 200)
    short_gold_total = sum(1 for r in rows for g in r["gold"]
                            if g in COURT_VOCAB and len(cit2text.get(g, "") or "") < 200)
    print(f"  gold courts that are substantive (>=200): {sub_gold_total} / {sub_gold_total+short_gold_total}")
    print(f"  gold courts that are short (<200) — would be lost if we drop short: {short_gold_total}")


if __name__ == "__main__":
    main()
