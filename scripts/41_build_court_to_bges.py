"""
C: build court → cited BGE graph.

Scan all 2M court_considerations rows. Use regex to find BGE references in the
text body (e.g. 'BGE 137 IV 122 E. 6.2', 'BGE 132 I 21 E. 3.2'). Match against
the BGE vocabulary (bge_only_doc_ids_v2.pkl) for exact citations.

Output: graph/court_to_bges_v1.pkl
   dict[court_citation] = [bge_citations]   # list of BGE citations cited by this court
"""
import os, re, time, pickle, json
from pathlib import Path
from collections import defaultdict

import polars as pl

P = Path("~/legal-ir").expanduser()
OUT_PATH = P / "graph/court_to_bges_v1.pkl"
STATS_PATH = P / "graph/court_to_bges_stats_v1.json"

BGE_DOC_IDS_PKL = P / "indexes/bge_only_doc_ids_v2.pkl"
COURT_PARQUET = P / "parquet/court_considerations.parquet"

# Match e.g.:
#   BGE 137 IV 122 E. 6.2
#   BGE 132 I 21 E. 3.2.2
#   BGE 145 II 32  (no E.)
BGE_REF_RE = re.compile(
    r"BGE\s*(\d+)\s+([IVX]+)\s+(\d+)"
    r"(?:\s*(?:E\.|Erw\.|consid\.|cons\.)\s*(\d+(?:\.\d+)*[a-zA-Z]?))?",
    re.IGNORECASE,
)


def main():
    if OUT_PATH.exists():
        print(f"[skip] {OUT_PATH} already exists")
        return
    t0 = time.time()
    with open(BGE_DOC_IDS_PKL, "rb") as f:
        bge_vocab = set(pickle.load(f))
    print(f"[load] BGE vocab: {len(bge_vocab):,}")

    print("[load] court_considerations.parquet")
    df = pl.read_parquet(COURT_PARQUET).select(["citation", "text"]).to_pandas()
    print(f"  rows: {len(df):,}")
    print(f"  load+convert: {time.time()-t0:.1f}s")

    court_to_bges = {}
    n_with = 0
    n_total_refs = 0
    n_refs_in_vocab = 0

    t1 = time.time()
    citations = df["citation"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    for i, (c, txt) in enumerate(zip(citations, texts)):
        if i % 200_000 == 0 and i > 0:
            elapsed = time.time() - t1
            rate = i / elapsed
            eta = (len(df) - i) / rate
            print(f"  {i:,}/{len(df):,}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  found_so_far={n_with:,}")
        # Find all BGE references in the text
        found = []
        for m in BGE_REF_RE.finditer(txt):
            num, vol, page, ercons = m.group(1), m.group(2).upper(), m.group(3), m.group(4)
            n_total_refs += 1
            # Construct candidate forms — try with E. then without
            base = f"BGE {num} {vol} {page}"
            if ercons:
                cand = f"{base} E. {ercons}"
                if cand in bge_vocab:
                    found.append(cand); n_refs_in_vocab += 1; continue
            # try without E.
            if base in bge_vocab:
                found.append(base); n_refs_in_vocab += 1
        if found:
            court_to_bges[c] = list(dict.fromkeys(found))   # dedup keep order
            n_with += 1

    print(f"\n[stats]")
    print(f"  courts with at least 1 BGE ref: {n_with:,} ({n_with/len(df)*100:.1f}%)")
    print(f"  total raw BGE references found: {n_total_refs:,}")
    print(f"  references matching vocab: {n_refs_in_vocab:,} ({n_refs_in_vocab/max(1,n_total_refs)*100:.1f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(court_to_bges, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    stats = {
        "courts_total": len(df),
        "courts_with_bge_ref": n_with,
        "raw_bge_refs": n_total_refs,
        "refs_in_vocab": n_refs_in_vocab,
        "build_time_sec": time.time() - t0,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"[saved] {STATS_PATH}")


if __name__ == "__main__":
    main()
