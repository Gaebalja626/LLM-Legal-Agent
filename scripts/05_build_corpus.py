"""
Build citation→text lookup for reranker stage.

Same dedup logic as PIPELINE_STATE.md so we get the same text the FAISS index
was built on:
  - laws + court_considerations merged
  - dedup on citation, text concat capped at 3000 chars
  - garbage rows (<30 chars) replaced by citation string
Output: ~/legal-ir/parquet/citation_text_v2.parquet
        + a pickle dict for fastest in-process lookup
"""
import os, time, gc, pickle
from pathlib import Path

import pandas as pd
import polars as pl

P = Path("~/legal-ir").expanduser()
PARQ = P / "parquet"
OUT_PARQ = PARQ / "citation_text_v2.parquet"
OUT_PKL = PARQ / "citation_text_v2.pkl"


def main():
    if OUT_PKL.exists():
        print(f"[skip] {OUT_PKL} exists; nothing to do")
        return

    t0 = time.time()
    print("[load] laws_de.parquet")
    laws = pd.read_parquet(PARQ / "laws_de.parquet")
    print(f"  laws: {len(laws):,}")

    print("[load] court_considerations.parquet (polars)")
    court = pl.read_parquet(PARQ / "court_considerations.parquet").to_pandas()
    print(f"  court: {len(court):,}")

    laws_clean = pd.DataFrame({
        "citation": laws["citation"],
        "text": laws["text"].astype(str),
        "source": "law",
    })
    court_clean = pd.DataFrame({
        "citation": court["citation"],
        "text": court["text"].astype(str),
        "source": "court",
    })
    del laws, court
    gc.collect()

    corpus = pd.concat([laws_clean, court_clean], ignore_index=True)
    del laws_clean, court_clean
    gc.collect()
    print(f"[merge] corpus rows: {len(corpus):,}")

    print("[dedup] groupby citation, concat-cap 3000")
    t = time.time()
    corpus = (
        corpus.groupby("citation", as_index=False)
              .agg({"text": lambda xs: " ".join(xs.astype(str))[:3000],
                    "source": "first"})
    )
    print(f"  done in {time.time() - t:.1f}s, unique citations: {len(corpus):,}")

    # garbage replace
    mask_short = corpus["text"].str.len() < 30
    print(f"[clean] short texts (<30 chars): {mask_short.sum():,}")
    corpus.loc[mask_short, "text"] = corpus.loc[mask_short, "citation"]

    # save parquet for inspection
    corpus.to_parquet(OUT_PARQ, compression="zstd")
    print(f"[saved] {OUT_PARQ} ({OUT_PARQ.stat().st_size/1e6:.1f} MB)")

    # save pickle dict for fast in-process lookup
    citation_to_text = dict(zip(corpus["citation"], corpus["text"]))
    with open(OUT_PKL, "wb") as f:
        pickle.dump(citation_to_text, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {OUT_PKL} ({OUT_PKL.stat().st_size/1e6:.1f} MB), entries={len(citation_to_text):,}")
    print(f"[done] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
