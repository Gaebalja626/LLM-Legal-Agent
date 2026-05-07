"""
EDA: BGE court text patterns.

1. Sample 500 BGEs and 200 random docket courts; classify the opening
   text pattern.
2. Compute pattern distribution on:
     - all BGE corpus
     - val gold BGEs only
     - random docket courts
3. Length stats per pattern.

Patterns (heuristic on first ~200 chars):
   - "statute_opening": starts with citation reference (Art./Nach Art./Gemäss Art./§/Conformément à l'art./En vertu de l'art./...)
   - "section_number_opening": starts with paragraph number like "3.1.", "4.1." etc.
   - "case_narrative":   starts with named subject ("A.________", "Der Beschwerdeführer", "Im vorliegenden Fall", "X. ist...", date, etc.)
   - "doctrinal_thesis": starts with abstract assertion ("Eine Eingrenzung...", "Kollusion bedeutet...", etc.)
   - "boilerplate": short cost/dispositif lines
   - "other"

For each gold BGE in val:
   show first 200 chars + classified pattern.
"""
import os, re, pickle, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

P = Path("~/legal-ir").expanduser()
CIT_PKL  = P / "parquet/citation_text_v2.pkl"


def classify(txt):
    if not txt: return "empty"
    t = txt.strip()
    if len(t) < 60:
        return "boilerplate"
    head = t[:250]
    head_lower = head.lower()

    # statute opening (DE, FR, IT)
    statute_starts = (
        "nach art", "gemäss art", "gemäß art", "gem. art", "art.", "§",
        "conformément à l'art", "selon l'art", "en vertu de l'art",
        "secondo l'art", "ai sensi dell'art", "in base all'art",
        "vorliegend ist art", "vorliegend ergibt sich aus art",
    )
    if any(t.lower().startswith(p) for p in statute_starts):
        return "statute_opening"

    # section number / Erwägung sub-paragraph
    if re.match(r"^\d+(\.\d+)*\.\s", t) or re.match(r"^\d+(\.\d+)*\s+[A-ZÄÖÜ]", t):
        return "section_number_opening"

    # case narrative — named subject (anonymized initials), dates, "Der Beschwerdeführer", "Im Fall"
    case_starts = (
        "im vorliegenden fall", "im fall", "der beschwerdeführer",
        "der beschwerdegegner", "die beschwerdeführerin", "die beschwerdegegnerin",
        "der angeklagte", "die parteien", "die vorinstanz", "das obergericht",
        "das bundesgericht", "le recourant", "la recourante", "le recourant ",
        "vorliegend", "x.", "y.", "a.", "b.", "c.",
    )
    if any(head_lower.startswith(p) for p in case_starts):
        return "case_narrative"
    # anonymized initial pattern like "A.________"
    if re.match(r"^[A-Z]\.[_]+", head):
        return "case_narrative"

    # doctrinal_thesis — starts with abstract noun + verb (best guess: capitalized noun followed by verb-like word)
    if re.match(r"^[A-ZÄÖÜ]\w+\s+(bedeutet|ist|sind|gilt|kommt|umfasst|wird|kann|muss|setzt voraus|liegt vor|verletzt)", head):
        return "doctrinal_thesis"

    return "other"


def main():
    print("[load]")
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)
    with open(P / "indexes/bge_only_doc_ids_v2.pkl", "rb") as f:
        BGE_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)

    val = pd.read_parquet(P / "parquet/val.parquet")
    val_gold_courts = []
    for vq in val.itertuples():
        for g in (vq.gold_citations or "").split(";"):
            g = g.strip()
            if g and g in set(COURT_DOC_IDS):
                val_gold_courts.append(g)

    BGE_VOCAB = set(BGE_DOC_IDS)
    val_gold_bges = [g for g in val_gold_courts if g in BGE_VOCAB]
    val_gold_dockets = [g for g in val_gold_courts if g not in BGE_VOCAB]
    print(f"  val gold courts: {len(val_gold_courts)} | BGE: {len(val_gold_bges)} | docket: {len(val_gold_dockets)}")

    rng = np.random.default_rng(42)

    # Sample 500 BGEs (random)
    bge_sample = [BGE_DOC_IDS[int(i)] for i in rng.choice(len(BGE_DOC_IDS), 500, replace=False)]
    bge_classes = Counter(classify(cit2text.get(c, "") or "") for c in bge_sample)
    print(f"\n[BGE random sample (500)] pattern distribution:")
    for k, n in bge_classes.most_common():
        print(f"  {k:<25}: {n} ({n/500*100:.1f}%)")

    # Classify val gold BGEs
    val_bge_classes = Counter(classify(cit2text.get(c, "") or "") for c in val_gold_bges)
    print(f"\n[val gold BGE ({len(val_gold_bges)})] pattern distribution:")
    for k, n in val_bge_classes.most_common():
        print(f"  {k:<25}: {n} ({n/max(1,len(val_gold_bges))*100:.1f}%)")

    # Classify val gold dockets
    val_docket_classes = Counter(classify(cit2text.get(c, "") or "") for c in val_gold_dockets)
    print(f"\n[val gold docket ({len(val_gold_dockets)})] pattern distribution:")
    for k, n in val_docket_classes.most_common():
        print(f"  {k:<25}: {n} ({n/max(1,len(val_gold_dockets))*100:.1f}%)")

    # Show 3 examples per pattern (val gold BGE)
    print(f"\n[val gold BGE examples per pattern]")
    by_pattern = {}
    for c in val_gold_bges:
        p = classify(cit2text.get(c, "") or "")
        by_pattern.setdefault(p, []).append(c)
    for p, items in sorted(by_pattern.items(), key=lambda x: -len(x[1])):
        print(f"\n  --- {p} ({len(items)}) ---")
        for c in items[:3]:
            head = (cit2text.get(c, "") or "")[:200].replace("\n", " ")
            print(f"    [{c}] {head}...")

    # Random docket sample
    docket_idx = [BGE_DOC_IDS[int(i)] for i in rng.choice(len(BGE_DOC_IDS), 1, replace=False)]  # placeholder
    # Pick non-BGE
    non_bge_courts = [c for c in COURT_DOC_IDS[:300_000] if c not in BGE_VOCAB][:500]  # cheap
    docket_classes = Counter(classify(cit2text.get(c, "") or "") for c in non_bge_courts)
    print(f"\n[random docket sample ({len(non_bge_courts)})] pattern distribution:")
    for k, n in docket_classes.most_common():
        print(f"  {k:<25}: {n} ({n/len(non_bge_courts)*100:.1f}%)")


if __name__ == "__main__":
    main()
