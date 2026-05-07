"""
Embedding-space alignment diagnostic.

For each val query × each multi-view variant, compute cosine similarity to
each gold citation's text (raw, and contextual-aug if available). Goals:

  1. Which query view best aligns with golds (per query, on average)?
  2. How much does cosine differ between gold-vs-non-gold neighbors?
  3. For which golds is the cosine just too low (manifold gap) regardless
     of which query view we use?

Output: a per-query view-alignment table + sorted gold-by-view scores.
"""
import os, json, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

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
    labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)
    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_VOCAB = set(pickle.load(f))
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_VOCAB = set(pickle.load(f))

    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    def embed(texts):
        return enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False, batch_size=32).astype("float32")

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

    # Aggregate per-view best/avg cosine to gold
    print("=" * 100)
    print("Per-query view→gold cosine alignment")
    print("=" * 100)
    print(f"{'qid':<8} {'view':<14} {'avg_cos_law':>12} {'avg_cos_court':>14} {'best_law':>10} {'best_court':>11}")
    overall_view_avg = defaultdict(list)
    for r in rows:
        gold_law = [g for g in r["gold"] if g in LAW_VOCAB]
        gold_court = [g for g in r["gold"] if g in COURT_VOCAB]
        # embed gold raw
        gold_law_emb = embed([cit2text.get(g, "") or g for g in gold_law]) if gold_law else None
        gold_court_emb = embed([cit2text.get(g, "") or g for g in gold_court]) if gold_court else None
        for view in MODES:
            v_text = r.get(view)
            if not v_text: continue
            qv = embed([v_text])
            if gold_law_emb is not None:
                cos_law = (qv @ gold_law_emb.T)[0]   # shape (n_gold,)
                avg_l = float(cos_law.mean())
                best_l = float(cos_law.max())
            else:
                avg_l = best_l = 0.0
            if gold_court_emb is not None:
                cos_c = (qv @ gold_court_emb.T)[0]
                avg_c = float(cos_c.mean())
                best_c = float(cos_c.max())
            else:
                avg_c = best_c = 0.0
            overall_view_avg[view].append((avg_l, avg_c))
            print(f"{r['qid']:<8} {view:<14} {avg_l:>12.4f} {avg_c:>14.4f} {best_l:>10.4f} {best_c:>11.4f}")
        print()

    print(f"\n{'='*100}")
    print("Average over all 10 queries (per view)")
    print(f"{'='*100}")
    print(f"  {'view':<16} {'avg_cos_law':>12} {'avg_cos_court':>14}")
    for view in MODES:
        items = overall_view_avg[view]
        if not items: continue
        avg_l = sum(x[0] for x in items) / len(items)
        avg_c = sum(x[1] for x in items) / len(items)
        print(f"  {view:<16} {avg_l:>12.4f} {avg_c:>14.4f}")

    # For one specific weak query (val_010), drill in: per-gold, per-view cosine
    print(f"\n{'='*100}")
    print("Drill: val_010 per-gold per-view cosine matrix")
    print(f"{'='*100}")
    r10 = next(r for r in rows if r["qid"] == "val_010")
    gold_text = [(g, cit2text.get(g, "") or g) for g in r10["gold"]]
    gold_emb = embed([t for _, t in gold_text])
    print(f"  query views: {MODES}")
    print(f"  gold count: {len(gold_text)}")
    # cosine matrix
    qv_list = []
    for view in MODES:
        v = r10.get(view)
        if v: qv_list.append((view, embed([v])[0]))
    print(f"\n  {'gold':<40} " + "  ".join(f"{v:<10}" for v, _ in qv_list))
    for (gid, _), emb in zip(gold_text, gold_emb):
        cos = []
        for view, qv in qv_list:
            cos.append(float(qv @ emb))
        cat = "L" if gid in LAW_VOCAB else ("C" if gid in COURT_VOCAB else "?")
        print(f"  [{cat}] {gid[:38]:<38} " + "  ".join(f"{c:>9.4f} " for c in cos))

    # If contextual labels exist, compare cosine raw vs aug for the BGEs that have labels
    if labels:
        print(f"\n{'='*100}")
        print("Compare raw-text vs aug-text gold embedding cosine for BGEs with labels")
        print(f"{'='*100}")
        bge_with_label = [(c, lab) for c, lab in labels.items() if lab.get("parsed")]
        print(f"  BGE pool with labels: {len(bge_with_label)}")
        # for val_001 query views, compare best cos to BGE in pool — raw vs aug
        r1 = next(r for r in rows if r["qid"] == "val_001")
        pool_cits = [c for c, _ in bge_with_label]
        raw_texts = [cit2text.get(c, "") or "" for c in pool_cits]
        aug_texts = [
            f"{lab['parsed'].get('context_en', '')} {lab['parsed'].get('context_de', '')} {cit2text.get(c, '') or ''}".strip()
            for c, lab in bge_with_label
        ]
        raw_emb = embed(raw_texts); aug_emb = embed(aug_texts)
        gold_in_pool = [g for g in r1["gold"] if g in {c for c, _ in bge_with_label}]
        # encode each query view and find the avg cosine to gold-in-pool
        print(f"\n  val_001 gold BGEs in pool: {len(gold_in_pool)}")
        gold_idx = [pool_cits.index(g) for g in gold_in_pool]
        for view in MODES:
            v = r1.get(view)
            if not v: continue
            qv = embed([v])[0]
            cos_raw = (qv @ raw_emb.T)[gold_idx]
            cos_aug = (qv @ aug_emb.T)[gold_idx]
            print(f"  view={view:<14} avg_cos_raw_gold={cos_raw.mean():.4f}  avg_cos_aug_gold={cos_aug.mean():.4f}  Δ={cos_aug.mean()-cos_raw.mean():+.4f}")


if __name__ == "__main__":
    main()
