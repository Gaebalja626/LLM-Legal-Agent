"""
Stage 9: build law-adjacency graph for citation neighbors expansion.

For each law citation in our LAW_DOC_IDS vocabulary, parse out
  (code, article_root, abs)
and group them. Two relations:
  R1 same_article: same (code, article_root) different abs → strong neighbors
  R2 adjacent_article: (code, art_n) ↔ (code, art_n±1) → weak neighbors

article_root: digits + optional letter suffix kept (so "11bis", "3a" are roots).
We do NOT walk arbitrary distance; only ±1 of the integer prefix in the root.

Output:
  graph/law_neighbors_v1.pkl
  dict[citation] = {
      "same_article": [neighbor citations],
      "adj_article" : [neighbor citations],
  }
"""
import os, re, time, pickle, json
from collections import defaultdict
from pathlib import Path

P = Path("~/legal-ir").expanduser()
OUT_PATH = P / "graph/law_neighbors_v1.pkl"
STATS_PATH = P / "graph/law_neighbors_stats_v1.json"

# Law citation regex matching the same shapes used elsewhere:
#   "Art. 11 ZGB"
#   "Art. 11 Abs. 2 OR"
#   "Art. 3a Abs. 1 lit. b StPO"
#   "Art. 11bis Abs. 2 BV"
LAW_RE = re.compile(
    r"^Art\.\s*"
    r"(?P<art>\d+[a-zA-Z]*(?:bis|ter|quater)?)"
    r"(?:\s*Abs\.\s*(?P<abs>\d+[a-zA-Z]*))?"
    r"(?:\s*lit\.\s*(?P<lit>[a-zA-Z]+))?"
    r"\s+(?P<code>[A-Za-z][\w\-\.]+)$"
)

NUM_RE = re.compile(r"^(\d+)([a-zA-Z]*(?:bis|ter|quater)?)?$")


def parse_law(cit):
    m = LAW_RE.match(cit.strip())
    if not m:
        return None
    art = m.group("art")
    abs_ = m.group("abs")
    lit = m.group("lit")
    code = m.group("code")
    nm = NUM_RE.match(art)
    if not nm:
        return None
    art_int = int(nm.group(1))
    art_suffix = nm.group(2) or ""
    return {
        "art_root": art,            # e.g. "11bis"
        "art_int": art_int,         # 11
        "art_suffix": art_suffix,   # "bis"
        "abs": abs_,
        "lit": lit,
        "code": code,
    }


def main():
    if OUT_PATH.exists():
        print(f"[skip] {OUT_PATH} already exists")
        return

    t0 = time.time()
    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    print(f"[load] LAW_DOC_IDS: {len(LAW_DOC_IDS):,}")

    parsed = {}
    parse_fail = 0
    for c in LAW_DOC_IDS:
        info = parse_law(c)
        if info is None:
            parse_fail += 1
        else:
            parsed[c] = info

    print(f"  parsed: {len(parsed):,} | parse_fail: {parse_fail:,}")

    # Group by (code, art_root) → all citations sharing the same article root
    same_article = defaultdict(list)
    by_code_art_int = defaultdict(list)   # (code, art_int) -> [(art_root, citation)]
    for c, info in parsed.items():
        same_article[(info["code"], info["art_root"])].append(c)
        by_code_art_int[(info["code"], info["art_int"])].append((info["art_root"], c))

    print(f"  unique (code, art_root) groups: {len(same_article):,}")
    print(f"  unique (code, art_int) groups: {len(by_code_art_int):,}")

    # Build neighbor map
    neighbors = {}
    for c, info in parsed.items():
        same = [x for x in same_article[(info["code"], info["art_root"])] if x != c]
        # adjacent integer ±1 within same code
        adj = []
        for delta in (-1, 1):
            for _root, cit in by_code_art_int.get((info["code"], info["art_int"] + delta), ()):
                adj.append(cit)
        neighbors[c] = {"same_article": same, "adj_article": adj}

    # quick stats
    same_lens = [len(v["same_article"]) for v in neighbors.values()]
    adj_lens = [len(v["adj_article"]) for v in neighbors.values()]
    avg_same = sum(same_lens) / max(1, len(same_lens))
    avg_adj = sum(adj_lens) / max(1, len(adj_lens))
    print(f"[stats] avg same_article neighbors: {avg_same:.2f}")
    print(f"[stats] avg adj_article neighbors:  {avg_adj:.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(neighbors, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    stats = {
        "total_citations": len(LAW_DOC_IDS),
        "parsed_ok": len(parsed),
        "parse_fail": parse_fail,
        "uniq_code_art_root_groups": len(same_article),
        "uniq_code_art_int_groups": len(by_code_art_int),
        "avg_same_article_neighbors": avg_same,
        "avg_adj_article_neighbors": avg_adj,
        "build_time_sec": time.time() - t0,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"[saved] {STATS_PATH}")
    print(f"[done] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
