"""
Stage 32: build law-law co-citation graph from court_to_laws.

For each court c with laws [L1, L2, ..., Ln], every pair (Li, Lj) gets +1.
Output: graph/law_cocite_v1.pkl
   dict[law] = list[(other_law, count)] sorted by count desc.

Plus stats.
"""
import os, time, pickle, json
from collections import defaultdict, Counter
from pathlib import Path

P = Path("~/legal-ir").expanduser()
OUT_PATH = P / "graph/law_cocite_v1.pkl"
STATS_PATH = P / "graph/law_cocite_stats_v1.json"


def main():
    if OUT_PATH.exists():
        print(f"[skip] {OUT_PATH} already exists")
        return
    t0 = time.time()
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    print(f"[load] court_to_laws: {len(court_to_laws):,}")

    pair_counts = Counter()
    n_pairs_total = 0
    for c, laws in court_to_laws.items():
        if len(laws) < 2:
            continue
        # unique laws within this court
        uniq = list(dict.fromkeys(laws))
        for i, a in enumerate(uniq):
            for b in uniq[i+1:]:
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1
                n_pairs_total += 2
    print(f"[count] unique directed pairs: {len(pair_counts):,} ({n_pairs_total:,} total)")
    print(f"[count] build time: {time.time()-t0:.1f}s")

    # Build dict[law] = list[(other, cnt)] sorted desc
    by_law = defaultdict(list)
    for (a, b), cnt in pair_counts.items():
        by_law[a].append((b, cnt))
    for a in by_law:
        by_law[a].sort(key=lambda x: -x[1])
        # cap to top-200 to keep file reasonable
        by_law[a] = by_law[a][:200]
    by_law = dict(by_law)

    # quick stats
    sizes = [len(v) for v in by_law.values()]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    print(f"[stats] laws covered: {len(by_law):,}")
    print(f"[stats] avg co-cite list size (capped 200): {avg_size:.1f}")
    print(f"[stats] max: {max_size}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(by_law, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    stats = {
        "courts_total": len(court_to_laws),
        "courts_with_2plus_laws": sum(1 for v in court_to_laws.values() if len(set(v)) >= 2),
        "directed_pairs": len(pair_counts),
        "laws_covered": len(by_law),
        "avg_neighbor_list_size": avg_size,
        "max_neighbor_list_size": max_size,
        "build_time_sec": time.time() - t0,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"[saved] {STATS_PATH}")


if __name__ == "__main__":
    main()
