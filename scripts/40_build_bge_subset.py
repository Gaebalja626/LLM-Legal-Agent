"""
A: build BGE-only subset index.

Reuses bge_m3_v2_clean_fp16.npy (full corpus embedding). Pulls out the rows
whose citation starts with 'BGE ' and writes a separate FAISS index +
matching doc_ids.
"""
import os, time, pickle, re
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

P = Path("~/legal-ir").expanduser()
EMB_PATH = P / "indexes/bge_m3_v2_clean_fp16.npy"
OUT_FAISS = P / "indexes/bge_only_v2.faiss"
OUT_DOC = P / "indexes/bge_only_doc_ids_v2.pkl"

BGE_RE = re.compile(r"^BGE\s")


def main():
    if OUT_FAISS.exists() and OUT_DOC.exists():
        print(f"[skip] {OUT_FAISS} + {OUT_DOC} already exist")
        return
    t0 = time.time()
    print("[load] corpus order (citation_text_v2.parquet) — same row order as bge_m3 npy")
    corpus = pd.read_parquet(P / "parquet/citation_text_v2.parquet")
    print(f"  rows: {len(corpus):,}")

    # We need the corpus order matching the .npy. The .npy was built from the SAME
    # corpus dataframe used in PIPELINE_STATE.md (groupby citation -> sorted by
    # whatever pandas gave). citation_text_v2.parquet was built with the same
    # groupby. Check it loads the same length.
    emb = np.load(EMB_PATH, mmap_mode="r")
    assert emb.shape[0] == len(corpus), \
        f"mismatch: emb has {emb.shape[0]} rows, corpus has {len(corpus)}"
    print(f"[ok] embedding shape: {emb.shape}")

    bge_mask = corpus["citation"].astype(str).str.match(BGE_RE).fillna(False).values
    n_bge = int(bge_mask.sum())
    print(f"[mask] BGE rows: {n_bge:,} ({n_bge/len(corpus)*100:.2f}%)")
    bge_doc_ids = corpus.loc[bge_mask, "citation"].tolist()
    bge_emb = emb[bge_mask].astype("float32")
    print(f"[emb] BGE-only embedding shape: {bge_emb.shape}")

    idx = faiss.IndexFlatIP(bge_emb.shape[1])
    idx.add(bge_emb)
    faiss.write_index(idx, str(OUT_FAISS))
    with open(OUT_DOC, "wb") as f:
        pickle.dump(bge_doc_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {OUT_FAISS} ({OUT_FAISS.stat().st_size/1e6:.1f} MB)")
    print(f"[saved] {OUT_DOC} ({OUT_DOC.stat().st_size/1e6:.1f} MB) — {len(bge_doc_ids):,} entries")
    print(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
