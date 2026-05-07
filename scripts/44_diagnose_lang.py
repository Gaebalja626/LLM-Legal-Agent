"""
Quick language diagnosis:
  1. corpus language distribution (heuristic word-based)
  2. val gold court language distribution
  3. our top-K retrieved court language distribution
  4. per-language hit rate

Heuristic: count distinctive function words. Fast (no model load).
"""
import os, json, pickle
from pathlib import Path
from collections import Counter, defaultdict

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
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")


DE_WORDS = {"der","die","das","und","ist","auf","von","den","dem","sich","mit","wird","nicht","auch",
            "art","abs","gemäss","gemäß","zur","zum","beim","eine","einer","eines","einen","durch"}
FR_WORDS = {"le","la","les","et","est","dans","des","pour","que","qui","une","aux","sur",
            "art","selon","par","au","du","ne","pas","plus","ainsi","ayant","cas"}
IT_WORDS = {"il","gli","della","del","che","per","una","non","con","sono","sul","alla","alle",
            "art","secondo","quanto","caso","stesso","dei","della","sulla"}


def detect_lang(text):
    if not text or len(text) < 10:
        return "?"
    s = text.lower()[:600]
    # crude tokenization — split on whitespace and strip punctuation
    toks = [t.strip(".,;:!?()[]{}«»\"'") for t in s.split()]
    de = sum(1 for t in toks if t in DE_WORDS)
    fr = sum(1 for t in toks if t in FR_WORDS)
    it = sum(1 for t in toks if t in IT_WORDS)
    if max(de, fr, it) == 0:
        return "?"
    if de >= fr and de >= it: return "de"
    if fr >= it: return "fr"
    return "it"


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]


def main():
    print("[load]")
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    COURT_VOCAB = set(COURT_DOC_IDS)

    # Step 1: corpus language distribution (sample 50k for speed; full takes longer)
    print("[step 1] corpus court language distribution (sample 100k)")
    sample_n = 100_000
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(len(COURT_DOC_IDS), size=sample_n, replace=False)
    counts = Counter()
    for idx in sample_ids:
        c = COURT_DOC_IDS[int(idx)]
        txt = cit2text.get(c, "") or ""
        counts[detect_lang(txt)] += 1
    print(f"  estimated language distribution (sample 100k):")
    total = sum(counts.values())
    for lang, n in counts.most_common():
        print(f"    {lang}: {n:>7,}  ({n/total*100:.1f}%)")
    print()

    # Step 2: val gold court languages
    print("[step 2] val gold court languages")
    val = pd.read_parquet(P / "parquet/val.parquet")
    gold_by_lang = defaultdict(list)
    for vq in val.itertuples():
        gold = split_cites(vq.gold_citations)
        for g in gold:
            if g in COURT_VOCAB:
                lang = detect_lang(cit2text.get(g, ""))
                gold_by_lang[lang].append((vq.query_id, g))
    for lang, items in gold_by_lang.items():
        print(f"  {lang}: {len(items)} gold courts")
        for q, g in items[:5]:
            preview = (cit2text.get(g, "") or "")[:80].replace("\n", " ")
            print(f"    [{q}] {g!r}: {preview}...")
        print()

    # Step 3: our top-K retrieval language distribution
    print("[step 3] our top-30 retrieved court language distribution (val)")
    val_exp = json.loads(EXP_PATH.read_text())
    mv = json.loads(MV_PATH.read_text())
    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))

    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256
    enc_cache = {}
    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]

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
    distinct = set()
    for r in rows:
        for k in MODES:
            if r.get(k): distinct.add(r[k])
    for q in distinct:
        encode(q)

    from collections import defaultdict as dd
    overall = Counter()
    per_q = {}
    for r in rows:
        # union top-30 over all 6 variants
        tops = set()
        for k in MODES:
            v = r.get(k)
            if v:
                cs = courts_for(v, 30)
                tops |= {d for d, _ in cs}
        c = Counter()
        for d in tops:
            c[detect_lang(cit2text.get(d, ""))] += 1
        per_q[r["qid"]] = c
        overall += c
    print(f"  overall (union top-30 from 6 variants × 10 queries):")
    for lang, n in overall.most_common():
        print(f"    {lang}: {n} ({n/sum(overall.values())*100:.1f}%)")
    print()
    print(f"  per-query (top-30 union):")
    for qid in val["query_id"].tolist():
        c = per_q[qid]
        line = f"    {qid}: "
        for lang in ("de","fr","it","?"):
            line += f"{lang}={c.get(lang,0):>3}  "
        print(line)

    # Step 4: per-language gold hit rate (using top-1000 union)
    print("\n[step 4] per-language gold hit rate (top-1000 union)")
    by_lang = {"de":[0,0], "fr":[0,0], "it":[0,0], "?":[0,0]}
    for r in rows:
        union = set()
        for k in MODES:
            v = r.get(k)
            if v:
                cs = courts_for(v, 1000)
                union |= {d for d, _ in cs}
        for g in r["gold"]:
            if g not in COURT_VOCAB: continue
            lang = detect_lang(cit2text.get(g, ""))
            by_lang[lang][1] += 1
            if g in union:
                by_lang[lang][0] += 1
    for lang in ("de","fr","it","?"):
        h, t = by_lang[lang]
        if t > 0:
            print(f"  {lang}: hit {h}/{t} = {h/t:.2f}")


if __name__ == "__main__":
    main()
