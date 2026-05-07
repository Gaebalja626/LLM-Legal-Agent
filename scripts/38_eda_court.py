"""
EDA: court_considerations corpus.
"""
import os, re, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

P = Path("~/legal-ir").expanduser()


def main():
    print("[load] court_considerations.parquet")
    df = pl.read_parquet(P / "parquet/court_considerations.parquet").to_pandas()
    print(f"  rows={len(df):,}  columns={list(df.columns)}")
    print(f"  dtypes:\n{df.dtypes}\n")

    # Length distribution
    text_lens = df["text"].astype(str).str.len()
    print(f"[length] text length stats:")
    print(f"  count: {len(text_lens):,}")
    print(f"  mean : {text_lens.mean():.0f}")
    print(f"  std  : {text_lens.std():.0f}")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        v = int(np.percentile(text_lens, q))
        print(f"  p{q:>2}  : {v:,}")
    print(f"  max  : {text_lens.max():,}")
    print()
    bins = [0, 100, 200, 300, 500, 1000, 2000, 5000, 10_000, 100_000]
    cuts = pd.cut(text_lens, bins, right=True)
    print(f"[length buckets]")
    print(cuts.value_counts().sort_index().to_string())
    print()

    # Citation patterns
    print("[citation] sample 30 citations")
    sample_cits = df["citation"].sample(30, random_state=0).tolist()
    for c in sample_cits:
        print(f"  {c!r}")
    print()

    # Pattern classification
    print("[citation pattern types]")
    BGE_re   = re.compile(r"^BGE\s")
    DOCK_re  = re.compile(r"^\d[A-Za-z]?[A-Z]?_\d+/\d{4}")           # e.g. 5A_800/2019
    DOCK_OLD = re.compile(r"^\d[A-Za-z]?\.\d+/\d{4}")                # e.g. 4P.260/2003
    UNS_re   = re.compile(r"\bUnpubliziert\b", re.I)
    types = Counter()
    for c in df["citation"].astype(str):
        if BGE_re.match(c):
            types["BGE"] += 1
        elif DOCK_re.match(c):
            types["docket_underscore (e.g. 5A_800/2019)"] += 1
        elif DOCK_OLD.match(c):
            types["docket_old_dot (e.g. 4P.260/2003)"] += 1
        else:
            types["other"] += 1
    for t, n in types.most_common():
        print(f"  {t:<35}: {n:>9,} ({n/len(df)*100:.2f}%)")
    print()

    # Show 10 random samples (citation + truncated text)
    print("[10 random samples]")
    samples = df.sample(10, random_state=42).reset_index(drop=True)
    for i, row in samples.iterrows():
        text = str(row["text"])
        print(f"\n--- sample {i+1} ---")
        print(f"citation: {row['citation']!r}")
        print(f"text len: {len(text)}")
        print(f"text[:600]: {text[:600]}")
        print(f"text[-200:]: ...{text[-200:]}")
        print()

    # Sample by length bucket: short/medium/long
    print("\n[by-length samples — 1 each from short/med/long]")
    for label, lo, hi in [("short", 0, 200), ("medium", 500, 1000), ("long", 2000, 5000), ("very long", 5000, 100000)]:
        sub = df[(text_lens >= lo) & (text_lens < hi)]
        if len(sub) == 0: continue
        s = sub.sample(1, random_state=7).iloc[0]
        print(f"\n--- {label} ({lo}-{hi}, n={len(sub):,}) ---")
        print(f"citation: {s['citation']!r}")
        print(f"text len: {len(str(s['text']))}")
        print(f"text[:400]: {str(s['text'])[:400]}")

    # Look at val_001's ground truth courts to see what they look like
    print("\n[val_001 gold court examples]")
    val = pd.read_parquet(P / "parquet/val.parquet")
    g = val.iloc[0]["gold_citations"].split(";")
    g = [x.strip() for x in g if x.strip()]
    courts_in_corpus = set(df["citation"])
    for c in g[:30]:
        if c in courts_in_corpus:
            row = df[df["citation"] == c].iloc[0]
            txt = str(row["text"])
            print(f"\n  {c!r}")
            print(f"    len: {len(txt)}")
            print(f"    [:300]: {txt[:300]}")

    # Citation column extras
    cols = list(df.columns)
    print(f"\n[other columns] besides citation/text:")
    for col in cols:
        if col not in ("citation", "text"):
            print(f"\n  {col}: dtype={df[col].dtype}, non-null={df[col].notna().sum():,}")
            sample_vals = df[col].dropna().sample(min(5, df[col].notna().sum()), random_state=0).tolist()
            for v in sample_vals:
                v_str = str(v)[:200]
                print(f"    {v_str!r}")


if __name__ == "__main__":
    main()
