"""
Diagnose why val_003 (0.028) and val_010 (0.040) score so low.

For each weak query:
  1. Print query and gold count + breakdown (laws vs courts; codes used).
  2. Check oracle: how many gold citations exist in corpus vocab?
  3. Check retrieval ceiling: how many gold are in our top-K candidates
     (top-200 law / top-200 court for each of orig/kw/hyde queries)?
  4. Show what our predicted top-15 laws + top-10 courts looked like.
  5. Show LLM expansion (codes, keywords).

This is a read-only inspection — no model loaded; only FAISS + vectors are
used because we already have the BGE-M3 cache from earlier runs is not
available in a separate process; so re-encode just for the 2 weak queries
plus val_001 as positive sanity.
"""
import os, json, pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

P = Path("~/legal-ir").expanduser()
TARGETS = ["val_003", "val_010", "val_007", "val_001"]   # weak first, then strong refs


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]


def main():
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads((P / "expansions/val_expansions_qwen3_14b.json").read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    LAW_VOCAB = set(LAW_DOC_IDS)
    COURT_VOCAB = set(COURT_DOC_IDS)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    with open(P / "parquet/citation_text_v2.pkl", "rb") as f:
        cit2text = pickle.load(f)

    print("[load] BGE-M3")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    K = 1000   # use a deep top-K to measure recall ceiling

    def encode(q):
        return enc.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype("float32")

    def topk(idx, q, doc_ids, k=K):
        v = encode(q)
        s, i = idx.search(v, k)
        return [(doc_ids[j], float(sc)) for j, sc in zip(i[0], s[0])]

    val_by_id = {r.query_id: r for r in val.itertuples()}

    for qid in TARGETS:
        print("\n" + "="*80)
        print(f"=== {qid} ===")
        print("="*80)
        row = val_by_id[qid]
        gold = split_cites(row.gold_citations)
        n_gold = len(gold)
        n_law_gold = sum(1 for g in gold if g in LAW_VOCAB)
        n_court_gold = sum(1 for g in gold if g in COURT_VOCAB)
        n_neither = sum(1 for g in gold if g not in LAW_VOCAB and g not in COURT_VOCAB)
        print(f"\nQuery (truncated):\n{row.query[:400].strip()}…\n")
        print(f"Gold count: {n_gold} (laws_in_vocab={n_law_gold}, courts_in_vocab={n_court_gold}, neither={n_neither})")
        if n_neither:
            missing = [g for g in gold if g not in LAW_VOCAB and g not in COURT_VOCAB]
            print(f"  ⚠ {len(missing)} gold citations NOT in any vocab. Examples: {missing[:5]}")

        exp = val_exp.get(qid, {}).get("parsed") or {}
        print(f"\nLLM expansion:")
        print(f"  legal_areas: {exp.get('legal_areas')}")
        print(f"  law_codes:   {exp.get('law_codes')}")
        kw = exp.get('keywords_de') or []
        print(f"  keywords_de ({len(kw)}): {kw[:8]}{'...' if len(kw)>8 else ''}")
        hy = exp.get('hyde') or ""
        print(f"  hyde[:200]: {hy[:200]}…")

        gold_law_codes = sorted({g.split()[-1] for g in gold if g in LAW_VOCAB})
        print(f"\nGold law codes: {gold_law_codes}")
        exp_codes = exp.get('law_codes') or []
        missing_codes = [c for c in gold_law_codes if c not in exp_codes]
        print(f"  Missing in expansion: {missing_codes}")

        # Retrieval ceiling per query variant
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde_q = exp.get("hyde") or None
        if hyde_q and len(hyde_q) < 50: hyde_q = None
        variants = [("orig", row.query), ("kw", kw_q), ("hyde", hyde_q)]

        print(f"\nRecall ceiling within top-{K} per query variant:")
        gold_set = set(gold)
        for vname, vq in variants:
            if not vq:
                print(f"  {vname}: -")
                continue
            laws = topk(law_index, vq, LAW_DOC_IDS, K)
            courts = topk(court_index, vq, COURT_DOC_IDS, K)
            law_hits = [(g, [d for d,_ in laws].index(g)+1) for g in gold if g in LAW_VOCAB and g in {d for d,_ in laws}]
            court_hits = [(g, [d for d,_ in courts].index(g)+1) for g in gold if g in COURT_VOCAB and g in {d for d,_ in courts}]
            print(f"  {vname}: law_hits={len(law_hits)}/{n_law_gold} (avg rank {np.mean([r for _,r in law_hits]):.0f} if any) | "
                  f"court_hits={len(court_hits)}/{n_court_gold}")
            if law_hits:
                print(f"      example law hits (rank): {law_hits[:5]}")

        # Court→law graph 1-hop ceiling
        # collect all laws cited by ANY of the gold court considerations
        gold_court_in_vocab = [g for g in gold if g in COURT_VOCAB]
        cite_pool = set()
        for g in gold_court_in_vocab:
            for law in court_to_laws.get(g, []):
                cite_pool.add(law)
        graph_law_hits = [g for g in gold if g in LAW_VOCAB and g in cite_pool]
        print(f"\nGraph 1-hop from gold courts → covers {len(graph_law_hits)}/{n_law_gold} gold laws")

        # Sample of corpus text for first gold (sanity that lookup works)
        if gold:
            first = gold[0]
            txt = cit2text.get(first, "<NOT FOUND>")
            print(f"\nFirst gold ({first}) text[:200]: {txt[:200]}…")


if __name__ == "__main__":
    main()
