"""
Stage 13: iterative agentic loop — round-2 expansion seeing round-1 hits.

Round 1: existing val_expansions_qwen3_14b.json + baseline pipeline → top-K1 laws + top-K1 courts.
Round 2: prompt LLM with the question AND the round-1 candidates (citations + short snippets) → ask for ADDITIONAL law_codes and keywords_de that fill gaps.
Final: union(r1, r2) expansion → re-run baseline pipeline on val.

Output:
  expansions/val_round2_qwen3_14b.json
  log table comparing F1 with vs without round 2.
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, pickle, time, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

P = Path("~/legal-ir").expanduser()
EXP_R1 = P / "expansions/val_expansions_qwen3_14b.json"
EXP_R2 = P / "expansions/val_round2_qwen3_14b.json"
CIT_PKL = P / "parquet/citation_text_v2.pkl"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05
ROUND1_LAW_SHOW = 25     # how many round-1 law candidates to show LLM
ROUND1_COURT_SHOW = 15
SNIPPET_CHARS = 220

ROUND2_SYSTEM = """You are a Swiss legal research assistant helping refine a retrieval pipeline.

You will see:
  - A legal question (English).
  - The first-round JSON expansion (law_codes / keywords / hyde).
  - The top candidates that retrieval returned in round 1, with short German snippets.

Your task:
  Identify what is MISSING. The retrieval likely missed important Swiss law abbreviations or specific German terms.
  Output a JSON with ADDITIONAL items only — no overlap with round 1 needed, no duplicates.

Output STRICTLY valid JSON:
{
  "additional_law_codes": [list of Swiss code abbreviations to add — focus on ones absent from round 1 candidates but plausibly relevant],
  "additional_keywords_de": [German legal phrases that would help find the missing articles — 5-15 items, specific],
  "refined_hyde": "A concise 100-200 word German legal note that focuses on the LIKELY GAPS (jurisdiction, procedure, related codes) — short and dense"
}

Output ONLY the JSON, nothing else.
"""


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]

def f1_per_query(p, g):
    p, g = set(p), set(g)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    tp = len(p & g)
    if tp == 0: return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)

def macro_f1(preds, golds):
    return float(np.mean([f1_per_query(p, g) for p, g in zip(preds, golds)]))

def rrf_merge(rankings, k=60):
    s = defaultdict(float)
    for r in rankings:
        for rank, (doc, _) in enumerate(r):
            s[doc] += 1.0 / (k + rank + 1)
    return s

def parse_json_loose(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    try:
        start = text.find("{")
        if start == -1: return None
        depth = 0; end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1; break
        if end == -1: return None
        return json.loads(text[start:end])
    except Exception:
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m: return json.loads(m.group(0))
        except Exception:
            pass
        return None


def main():
    print("[load] data + round-1 expansions")
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_R1.read_text())

    with open(P / "indexes/law_doc_ids_v2.pkl", "rb") as f:
        LAW_DOC_IDS = pickle.load(f)
    with open(P / "indexes/court_doc_ids_v2.pkl", "rb") as f:
        COURT_DOC_IDS = pickle.load(f)
    law_index = faiss.read_index(str(P / "indexes/law_v2.faiss"))
    court_index = faiss.read_index(str(P / "indexes/court_v2.faiss"))
    with open(P / "graph/court_to_laws_v1.pkl", "rb") as f:
        court_to_laws = pickle.load(f)
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)

    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    enc_cache, law_cache, court_cache = {}, {}, {}
    def encode(q):
        if q not in enc_cache:
            enc_cache[q] = enc.encode([q], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype("float32")
        return enc_cache[q]
    def laws_for(q):
        if q not in law_cache:
            v = encode(q); s, i = law_index.search(v, K_LAW_RAW)
            law_cache[q] = [(LAW_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return law_cache[q]
    def courts_for(q):
        if q not in court_cache:
            v = encode(q); s, i = court_index.search(v, K_COURT_RAW)
            court_cache[q] = [(COURT_DOC_IDS[idx], float(sc)) for idx, sc in zip(i[0], s[0])]
        return court_cache[q]

    def build_row(qid, query, exp_r1, exp_r2):
        kws = list(exp_r1.get("keywords_de") or [])
        codes = list(exp_r1.get("law_codes") or [])
        hyde = exp_r1.get("hyde") or ""
        # union round 2
        for c in (exp_r2.get("additional_law_codes") or []):
            if c not in codes: codes.append(c)
        for k in (exp_r2.get("additional_keywords_de") or []):
            if k not in kws: kws.append(k)
        hyde_extra = exp_r2.get("refined_hyde") or ""
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        # dedup keywords
        seen = set(); kws2 = []
        for k in kws:
            if k not in seen:
                seen.add(k); kws2.append(k)
        kws = kws2
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        return {
            "qid": qid,
            "orig": query,
            "kw": kw_q,
            "hyde": hyde or None,
            "hyde2": hyde_extra or None,
        }

    def predict(r, n_law=15, n_court=10, use_hyde2=True):
        rs_law, rs_court = [], []
        for k in ("orig","kw","hyde"):
            if r.get(k):
                rs_law.append(laws_for(r[k]))
                rs_court.append(courts_for(r[k]))
        if use_hyde2 and r.get("hyde2"):
            rs_law.append(laws_for(r["hyde2"]))
            rs_court.append(courts_for(r["hyde2"]))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:n_law]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:n_court]]

    # ---------- Round 1 candidates per query ----------
    base_rows = []
    for vq in val.itertuples():
        exp = val_exp.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde = exp.get("hyde") or None
        if hyde and len(hyde) < 50: hyde = None
        base_rows.append({"qid": vq.query_id, "orig": vq.query, "kw": kw_q, "hyde": hyde,
                          "exp_r1": exp,
                          "gold": split_cites(vq.gold_citations)})

    distinct = set()
    for r in base_rows:
        for k in ("orig","kw","hyde"):
            if r.get(k): distinct.add(r[k])
    print(f"[search] pre-warm {len(distinct)} round-1 queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    def round1_candidates(r):
        return predict({"qid": r["qid"], "orig": r["orig"], "kw": r["kw"], "hyde": r["hyde"]},
                       n_law=ROUND1_LAW_SHOW, n_court=ROUND1_COURT_SHOW, use_hyde2=False)

    print("[round1] computing candidates")
    r1_cands = {}
    for r in base_rows:
        cits = round1_candidates(r)
        r1_cands[r["qid"]] = cits

    # ---------- Round 2 expansion via LLM ----------
    if EXP_R2.exists():
        round2 = json.loads(EXP_R2.read_text())
        print(f"[load] existing round-2 expansions: {len(round2)} entries")
    else:
        round2 = {}

    todo = [r["qid"] for r in base_rows
            if round2.get(r["qid"], {}).get("parsed") is None]

    if todo:
        print(f"[load] {LLM_MODEL} for round-2 generation ({len(todo)} todo)")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, dtype=torch.float16, device_map="cuda")
        model.eval()
        print(f"  loaded in {time.time()-t0:.1f}s")

        for r in base_rows:
            qid = r["qid"]
            if qid not in todo: continue
            cits = r1_cands[qid]
            # build a candidate display string
            display = []
            laws_only = [c for c in cits if any(c.startswith(p) for p in ["Art."])]
            courts_only = [c for c in cits if c not in laws_only]
            for c in cits:
                txt = (cit2text.get(c, "") or "")[:SNIPPET_CHARS].replace("\n", " ")
                display.append(f"  - {c}: {txt}")
            display_str = "\n".join(display)
            user_msg = (
                f"Legal question:\n{r['orig']}\n\n"
                f"Round-1 expansion (already used):\n"
                f"  law_codes: {r['exp_r1'].get('law_codes')}\n"
                f"  keywords_de: {r['exp_r1'].get('keywords_de')}\n\n"
                f"Round-1 retrieval top candidates (German snippets):\n{display_str}\n\n"
                f"Now produce the JSON described in the system prompt."
            )
            msgs = [
                {"role": "system", "content": ROUND2_SYSTEM},
                {"role": "user",   "content": user_msg},
            ]
            try:
                prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok([prompt], return_tensors="pt", truncation=True, max_length=8192).to("cuda")
            t = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=1500, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_json_loose(text)
            round2[qid] = {"raw": text, "parsed": parsed}
            tag = "OK" if parsed else "FAIL"
            extra_codes = (parsed or {}).get("additional_law_codes", "?")
            extra_kws = (parsed or {}).get("additional_keywords_de", [])
            print(f"  [{qid}] {tag} ({time.time()-t:.1f}s): codes+={extra_codes}  kw+={len(extra_kws) if isinstance(extra_kws, list) else '?'}")
            EXP_R2.write_text(json.dumps(round2, ensure_ascii=False, indent=2))

        # free LLM
        del model
        torch.cuda.empty_cache()

    # ---------- Re-run search with combined expansion ----------
    print("\n[search] running combined (round1 + round2) pipeline")
    rows_combined = []
    for r in base_rows:
        exp_r2 = round2.get(r["qid"], {}).get("parsed") or {}
        rows_combined.append(build_row(r["qid"], r["orig"], r["exp_r1"], exp_r2))
        # remember gold
        rows_combined[-1]["gold"] = r["gold"]

    new_qs = set()
    for r in rows_combined:
        for k in ("orig","kw","hyde","hyde2"):
            if r.get(k): new_qs.add(r[k])
    new_qs -= set(law_cache.keys())
    if new_qs:
        t = time.time()
        for q in new_qs:
            laws_for(q); courts_for(q)
        print(f"  pre-warm {len(new_qs)} new queries in {time.time()-t:.1f}s")

    golds = [r["gold"] for r in rows_combined]

    # baseline reference
    print("\n[ref] baseline (round 1 only):")
    base_preds = []
    for r in base_rows:
        bp = predict({"qid": r["qid"], "orig": r["orig"], "kw": r["kw"], "hyde": r["hyde"]},
                     n_law=15, n_court=10, use_hyde2=False)
        base_preds.append(bp)
    print(f"  F1 = {macro_f1(base_preds, golds):.4f}")

    print("\n[ablation] round-1 + round-2 union")
    print(f"{'config':<28} {'F1':>8}  per-q {'val_001':>8}{'val_003':>8}{'val_007':>8}{'val_010':>8}")
    for nL, nC in [(15,10),(20,12),(25,15),(20,10),(25,10),(30,10)]:
        preds = [predict(r, n_law=nL, n_court=nC, use_hyde2=True) for r in rows_combined]
        f1 = macro_f1(preds, golds)
        per_q = {qid: f1_per_query(p, g) for qid, p, g in
                 zip([r["qid"] for r in rows_combined], preds, golds)}
        line = f"  union  nL={nL:>2} nC={nC:>2}        {f1:>8.4f}  "
        for qid in ["val_001","val_003","val_007","val_010"]:
            line += f"{per_q[qid]:>8.3f}"
        print(line)

    # also try without hyde2 (to test if just code/kw union helps)
    print()
    for nL, nC in [(15,10),(25,10)]:
        preds = [predict(r, n_law=nL, n_court=nC, use_hyde2=False) for r in rows_combined]
        f1 = macro_f1(preds, golds)
        per_q = {qid: f1_per_query(p, g) for qid, p, g in
                 zip([r["qid"] for r in rows_combined], preds, golds)}
        line = f"  no-hyde2 nL={nL:>2} nC={nC:>2}      {f1:>8.4f}  "
        for qid in ["val_001","val_003","val_007","val_010"]:
            line += f"{per_q[qid]:>8.3f}"
        print(line)


if __name__ == "__main__":
    main()
