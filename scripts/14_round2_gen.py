"""
Stage 14: round-2 expansion generation with repetition_penalty fix.

Modes:
  python 14_round2_gen.py val_retry   # only failed val entries
  python 14_round2_gen.py test        # all test queries
  python 14_round2_gen.py all         # both

Round-1 candidates are recomputed via the cached pipeline. LLM call uses
repetition_penalty=1.1 to avoid the "ZGB ZGB ZGB" runaway seen in 13.
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, pickle, time, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

P = Path("~/legal-ir").expanduser()
EXP_R1_VAL = P / "expansions/val_expansions_qwen3_14b.json"
EXP_R1_TEST = P / "expansions/test_expansions_qwen3_14b.json"
EXP_R2_VAL = P / "expansions/val_round2_qwen3_14b.json"
EXP_R2_TEST = P / "expansions/test_round2_qwen3_14b.json"
CIT_PKL = P / "parquet/citation_text_v2.pkl"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05
ROUND1_LAW_SHOW = 25
ROUND1_COURT_SHOW = 15
SNIPPET_CHARS = 220

ROUND2_SYSTEM = """You are a Swiss legal research assistant helping refine a retrieval pipeline.

You will see:
  - A legal question (English).
  - The first-round JSON expansion (law_codes / keywords / hyde).
  - The top candidates that retrieval returned in round 1, with short German snippets.

Your task:
  Identify what is MISSING. Retrieval likely missed important Swiss law abbreviations or specific German terms.
  Output a JSON with ADDITIONAL items only.

Output STRICTLY valid JSON, exact schema:
{
  "additional_law_codes": [Swiss code abbreviations to add — focus on codes plausibly relevant but absent from round 1, max 8],
  "additional_keywords_de": [German legal phrases to add, 5-12 items, specific],
  "refined_hyde": "A 100-200 word German legal note focusing on likely retrieval gaps."
}

Output ONLY the JSON, nothing else.
"""


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]

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
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert mode in ("val_retry", "test", "all")

    print("[load] indexes + corpus")
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

    print("[load] BGE-M3 (for round-1 candidates)")
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

    def round1_candidates(query, exp):
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        kw_q = " ".join(codes + kws) if (kws or codes) else None
        hyde = exp.get("hyde") or None
        if hyde and len(hyde) < 50: hyde = None
        rs_law, rs_court = [], []
        for q in [query, kw_q, hyde]:
            if q:
                rs_law.append(laws_for(q)); rs_court.append(courts_for(q))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        law_top = sorted(ls.items(), key=lambda x: -x[1])[:ROUND1_LAW_SHOW]
        return [d for d, _ in law_top] + [d for d, _ in court_top[:ROUND1_COURT_SHOW]]

    # determine work to do
    work = []  # list of (which, qid, query, exp_r1, target_path)
    if mode in ("val_retry", "all"):
        val = pd.read_parquet(P / "parquet/val.parquet")
        val_r1 = json.loads(EXP_R1_VAL.read_text())
        val_r2 = {}
        if EXP_R2_VAL.exists():
            val_r2 = json.loads(EXP_R2_VAL.read_text())
        for vq in val.itertuples():
            qid = vq.query_id
            if val_r2.get(qid, {}).get("parsed") is None:
                exp_r1 = val_r1.get(qid, {}).get("parsed") or {}
                work.append(("val", qid, vq.query, exp_r1, EXP_R2_VAL))

    if mode in ("test", "all"):
        test = pd.read_parquet(P / "parquet/test.parquet")
        test_r1 = json.loads(EXP_R1_TEST.read_text())
        test_r2 = {}
        if EXP_R2_TEST.exists():
            test_r2 = json.loads(EXP_R2_TEST.read_text())
        for tq in test.itertuples():
            qid = tq.query_id
            if test_r2.get(qid, {}).get("parsed") is None:
                exp_r1 = test_r1.get(qid, {}).get("parsed") or {}
                work.append(("test", qid, tq.query, exp_r1, EXP_R2_TEST))

    print(f"[work] todo: {len(work)} queries")
    if not work:
        print("[done] nothing to do")
        return

    # warm queries
    distinct = set()
    for _, _, q, exp, _ in work:
        distinct.add(q)
        kw_q = " ".join((exp.get("law_codes") or []) + (exp.get("keywords_de") or []))
        if kw_q: distinct.add(kw_q)
        hyde = exp.get("hyde") or ""
        if len(hyde) >= 50: distinct.add(hyde)
    distinct -= set(law_cache.keys())
    print(f"[warm] {len(distinct)} queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    print(f"[load] {LLM_MODEL}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # cache loaded files for incremental save
    saved = {}  # path -> dict
    for w in work:
        path = w[4]
        if path not in saved:
            saved[path] = json.loads(path.read_text()) if path.exists() else {}

    n_ok, n_fail = 0, 0
    for which, qid, query, exp_r1, target in work:
        cits = round1_candidates(query, exp_r1)
        display = []
        for c in cits:
            txt = (cit2text.get(c, "") or "")[:SNIPPET_CHARS].replace("\n", " ")
            display.append(f"  - {c}: {txt}")
        display_str = "\n".join(display)
        user_msg = (
            f"Legal question:\n{query}\n\n"
            f"Round-1 expansion (already used):\n"
            f"  law_codes: {exp_r1.get('law_codes')}\n"
            f"  keywords_de: {exp_r1.get('keywords_de')}\n\n"
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
            out = model.generate(
                **inputs, max_new_tokens=1500, do_sample=False,
                repetition_penalty=1.1,                 # <-- fix
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_json_loose(text)
        saved[target][qid] = {"raw": text, "parsed": parsed}
        if parsed: n_ok += 1
        else: n_fail += 1
        extra_codes = (parsed or {}).get("additional_law_codes", "?")
        extra_kws = (parsed or {}).get("additional_keywords_de", [])
        nk = len(extra_kws) if isinstance(extra_kws, list) else "?"
        print(f"  [{which}/{qid}] {'OK' if parsed else 'FAIL'} ({time.time()-t:.1f}s): codes+={extra_codes} kw+={nk}")
        target.write_text(json.dumps(saved[target], ensure_ascii=False, indent=2))

    print(f"\n[done] OK={n_ok} FAIL={n_fail}")


if __name__ == "__main__":
    main()
