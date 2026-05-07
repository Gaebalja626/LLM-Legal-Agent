"""
Stage 30: iterative rejection + replenishment loop (사용자 새 아이디어).

Round 1:
  - Best retrieval: top-K_show (15 law + 15 court) candidates
  - LLM call: for each candidate, verdict ∈ {keep, reject, uncertain}; also
    output 'missing' citations not in pool.
  - Apply:
      kept   = candidates with verdict='keep'
      pool   = kept ∪ missing
      replenish from retrieval ranks K_show+1 ... K_show*2 if pool too short

Round 2:
  - LLM call on new pool → verdict again → final
  - Final = round-2 keep ∪ round-2 missing, truncated to target_size

Variants tested:
  - 1-round (round-1 only)
  - 2-round
  - vary target_size

Saves verdict cache to expansions/val_rejection_loop_qwen3_14b.json
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
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
MV_PATH  = P / "expansions/val_multiview_qwen3_14b.json"
CIT_PKL  = P / "parquet/citation_text_v2.pkl"
OUT_PATH = P / "expansions/val_rejection_loop_qwen3_14b.json"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05
SHOW_N_LAW = 15
SHOW_N_COURT = 15
SNIPPET_CHARS = 200
MAX_NEW = 2000
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")

REJECT_SYSTEM = """You are a Swiss legal research assistant. You see an English Swiss-law question and a list of retrieval candidates (citations + German snippets). For each candidate, judge whether it is RELEVANT to the answer (keep), UNRELATED (reject), or UNCLEAR (uncertain). Be strict: if the snippet is on a different legal topic, reject it.

Also propose any MISSING Swiss-law citations that should be in the final answer but are not in the candidates.

Output STRICTLY one JSON object:
{
  "verdicts": [
    {"citation": "<copy verbatim>", "verdict": "keep" | "reject" | "uncertain"}
  ],
  "missing": [list of additional citations not in candidates; max 15; strict format 'Art. N Abs. M XYZ' or 'BGE V S P E. N']
}

Output ONLY the JSON, no extra text."""


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
        depth = 0; end = -1; in_str = False; esc = False
        for i, ch in enumerate(text[start:], start):
            if esc: esc = False; continue
            if ch == "\\": esc = True; continue
            if ch == '"': in_str = not in_str; continue
            if in_str: continue
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
        except Exception: pass
        return None


def main():
    print("[load]")
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_PATH.read_text())
    mv = json.loads(MV_PATH.read_text())
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
    distinct = set()
    for r in rows:
        for k in MODES:
            if r.get(k): distinct.add(r[k])
    print(f"[search] pre-warm {len(distinct)}")
    for q in distinct:
        laws_for(q); courts_for(q)

    def stage2(r):
        rs_law, rs_court = [], []
        for k in MODES:
            v = r.get(k)
            if v:
                rs_law.append(laws_for(v))
                rs_court.append(courts_for(v))
        ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
        court_top = sorted(cs.items(), key=lambda x: -x[1])
        gf = Counter()
        for d, _ in court_top[:GRAPH_POOL]:
            for law in court_to_laws.get(d, []):
                gf[law] += 1
        for law, freq in gf.items():
            ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
        law_top = sorted(ls.items(), key=lambda x: -x[1])
        return law_top, court_top

    rankings = [stage2(r) for r in rows]
    golds = [r["gold"] for r in rows]

    # ---------- Round 1 candidates per query ----------
    pools = {}
    for r, (law_top, court_top) in zip(rows, rankings):
        cands = ([d for d, _ in law_top[:SHOW_N_LAW]]
                 + [d for d, _ in court_top[:SHOW_N_COURT]])
        # Reserve replenishment candidates
        replen = ([d for d, _ in law_top[SHOW_N_LAW:SHOW_N_LAW*2]]
                  + [d for d, _ in court_top[SHOW_N_COURT:SHOW_N_COURT*2]])
        pools[r["qid"]] = {"cands": cands, "replen": replen,
                           "law_top": law_top, "court_top": court_top}

    # ---------- LLM call helper ----------
    cache = {}
    if OUT_PATH.exists():
        cache = json.loads(OUT_PATH.read_text())

    tok = None; model = None

    def llm_verdict(qid_round_key, query, candidates):
        if qid_round_key in cache and cache[qid_round_key].get("parsed"):
            return cache[qid_round_key]
        nonlocal tok, model
        if model is None:
            print(f"[load] {LLM_MODEL}")
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(LLM_MODEL)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL, dtype=torch.float16, device_map="cuda")
            model.eval()
            print(f"  loaded in {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

        display = []
        for c in candidates:
            txt = (cit2text.get(c, "") or "")[:SNIPPET_CHARS].replace("\n", " ")
            display.append(f"  - {c}: {txt}")
        user_msg = (
            f"Question (English):\n{query}\n\n"
            f"Candidates ({len(candidates)}):\n" + "\n".join(display) +
            "\n\nProduce the JSON now."
        )
        msgs = [
            {"role": "system", "content": REJECT_SYSTEM},
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
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 repetition_penalty=1.05,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_json_loose(text)
        cache[qid_round_key] = {"raw": text, "parsed": parsed}
        OUT_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
        n_v = len((parsed or {}).get("verdicts", []) or [])
        n_keep = sum(1 for v in (parsed or {}).get("verdicts", []) if v.get("verdict") == "keep")
        n_rej = sum(1 for v in (parsed or {}).get("verdicts", []) if v.get("verdict") == "reject")
        n_miss = len((parsed or {}).get("missing", []) or [])
        print(f"  {qid_round_key} ({time.time()-t:.1f}s): verdicts={n_v} (keep={n_keep} rej={n_rej}) miss={n_miss}")
        return cache[qid_round_key]

    # ---------- Run round 1 ----------
    print("\n[round 1] LLM verdicts on top-30 candidates")
    round1 = {}
    for r in rows:
        qid = r["qid"]
        cands = pools[qid]["cands"]
        v = llm_verdict(f"{qid}__r1", r["orig"], cands)
        round1[qid] = v.get("parsed") or {}

    # Apply round-1 verdicts → new candidate pool for round 2
    print("\n[apply] removing rejected, adding missing, replenishing")
    pool2 = {}
    for r in rows:
        qid = r["qid"]
        cands = pools[qid]["cands"]
        replen = pools[qid]["replen"]
        verd = round1.get(qid) or {}
        verdicts = verd.get("verdicts") or []
        rejected = {v.get("citation") for v in verdicts if v.get("verdict") == "reject"}
        kept = [c for c in cands if c not in rejected]
        missing = [c for c in (verd.get("missing") or []) if c]
        new_pool = []
        seen = set()
        for c in kept + missing:
            if c not in seen:
                seen.add(c); new_pool.append(c)
        # replenish back to original size
        for c in replen:
            if len(new_pool) >= len(cands):
                break
            if c not in seen:
                seen.add(c); new_pool.append(c)
        pool2[qid] = new_pool

    # ---------- Run round 2 ----------
    print("\n[round 2] LLM verdicts on refreshed pool")
    round2 = {}
    for r in rows:
        qid = r["qid"]
        v = llm_verdict(f"{qid}__r2", r["orig"], pool2[qid])
        round2[qid] = v.get("parsed") or {}

    # ---------- Build final variants ----------
    def build_final(qid, source, target_size):
        if source == "round1_keep_plus_missing":
            verd = round1.get(qid) or {}
            cands = pools[qid]["cands"]
            verdicts = verd.get("verdicts") or []
            rejected = {v.get("citation") for v in verdicts if v.get("verdict") == "reject"}
            kept = [c for c in cands if c not in rejected]
            missing = [c for c in (verd.get("missing") or []) if c]
            seen = set(); out = []
            for c in kept + missing:
                if c not in seen:
                    seen.add(c); out.append(c)
        elif source == "round2_keep_plus_missing":
            verd = round2.get(qid) or {}
            cands2 = pool2[qid]
            verdicts = verd.get("verdicts") or []
            rejected = {v.get("citation") for v in verdicts if v.get("verdict") == "reject"}
            kept = [c for c in cands2 if c not in rejected]
            missing = [c for c in (verd.get("missing") or []) if c]
            seen = set(); out = []
            for c in kept + missing:
                if c not in seen:
                    seen.add(c); out.append(c)
        elif source == "intersection_r1_r2":
            verd1 = round1.get(qid) or {}
            verd2 = round2.get(qid) or {}
            cands = pools[qid]["cands"]
            v1 = verd1.get("verdicts") or []
            rejected1 = {v.get("citation") for v in v1 if v.get("verdict") == "reject"}
            kept1 = [c for c in cands if c not in rejected1]
            v2 = verd2.get("verdicts") or []
            rejected2 = {v.get("citation") for v in v2 if v.get("verdict") == "reject"}
            cands2 = pool2[qid]
            kept2 = [c for c in cands2 if c not in rejected2]
            both = [c for c in kept1 if c in set(kept2)]
            missing = list((verd1.get("missing") or []) + (verd2.get("missing") or []))
            seen = set(); out = []
            for c in both + kept1 + kept2 + missing:
                if c not in seen:
                    seen.add(c); out.append(c)
        else:
            raise ValueError(source)
        return out[:target_size]

    print("\n[eval] rejection-loop variants")
    print(f"  reference: nL=15 nC=0 base = 0.2064; base+add(t=17) = 0.2088")
    for src in ["round1_keep_plus_missing", "round2_keep_plus_missing", "intersection_r1_r2"]:
        print(f"\n  source={src}")
        for ts in [12, 14, 15, 16, 17, 18, 20, 22, 25]:
            preds = [build_final(r["qid"], src, ts) for r in rows]
            f1 = macro_f1(preds, golds)
            per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
            print(f"    target={ts}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
