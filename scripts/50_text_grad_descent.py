"""
Text gradient descent on multi-view expansion prompt.

Loop:
  1. With current SYSTEM prompt, generate val multi-view expansions.
  2. Run retrieval (current best pipeline) + measure F1 + per-query weaknesses.
  3. Send (prompt, score, per-query weak examples, missed gold) to a critic LLM
     and ask for a revised prompt.
  4. Accept revised prompt if F1 improves, else revert.
  5. Repeat for max_iter or until convergence.

Outputs:
  expansions/textgrad_iter_<N>_expansions.json  (val expansions per iter)
  expansions/textgrad_history.json              (prompt + score history)
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, pickle, time, re, sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
CIT_PKL  = P / "parquet/citation_text_v2.pkl"
HIST_PATH = P / "expansions/textgrad_history.json"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05
N_LAW = 15
N_COURT = 0    # we already know nC=0 is best
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")

INITIAL_SYSTEM = """You are a Swiss legal research assistant. Given an English Swiss-law question, produce SIX different German views useful for retrieval over a German legal corpus.

Output STRICTLY a single JSON object with these keys (all in German except where noted):

{
  "trans_natural":  "Faithful natural-German translation of the question, preserving every fact, date, citation. Mid-length, conversational legal register.",
  "trans_formal":   "Same content but in formal Swiss legal-document register (e.g. 'Es stellt sich die Frage, ob ...'). Avoid colloquialisms.",
  "trans_concise":  "Compressed German version: only the essential facts and legal issues, max 60 words. Strip narrative details.",
  "hyde_statute":   "A 100-200 word German answer in the style of a STATUTE / commentary — abstract, normative, citing 'Art. N Abs. M XYZ' references. Match how Swiss law articles are written.",
  "hyde_case":      "A 100-200 word German answer in the style of a Federal Tribunal CASE consideration ('Erwägung') — narrative, fact-applying, with 'BGE'/docket-style references when natural.",
  "sub_questions":  ["3 separate German sub-questions, each focused on ONE legal sub-issue raised by the original question. Each 1-2 sentences."]
}

Rules:
- All German, no English (except citations like 'Art.', 'BGE').
- Use precise Swiss legal terminology (Untersuchungshaft, Verhältnismässigkeit, etc.) — NOT word-for-word English.
- No duplicates between views.
- Output ONLY the JSON object — no markdown fences, no preamble, no extra text.
"""

CRITIC_SYSTEM = """You are an expert prompt engineer optimizing a Swiss-law retrieval system.

The current SYSTEM prompt asks an LLM to produce 6 German views of an English Swiss-law question. These views are then embedded with a multilingual encoder and used to retrieve from a German Swiss legal corpus.

You will see:
  - The current SYSTEM prompt.
  - The current Macro F1 on validation (target: maximize).
  - Per-query F1 breakdown.
  - For weak queries: examples of MISSED gold citations (laws and courts that the system failed to retrieve), and the actual expansion produced.

Your task: propose a REVISED SYSTEM prompt that should improve retrieval.

Strategies you may consider:
  - Force more specific Swiss legal terminology (e.g. exact statute abbreviations like StPO, OR, ZGB, IPRG, SchKG; doctrinal terms like Verhältnismässigkeit, Kollusionsgefahr).
  - Match the writing style of Swiss BGE court considerations (abstract doctrinal rule statements, statute citations).
  - Cover French/Italian legal terminology when relevant (Swiss corpus is multi-language).
  - Add a view that explicitly lists likely cited articles in 'Art. N Abs. M XYZ' form.
  - Tighten or change the structure of any view.

Output STRICTLY one JSON:
{
  "rationale": "<2-3 sentence reasoning for the changes>",
  "revised_system_prompt": "<the full revised prompt to use as SYSTEM>"
}

Output ONLY the JSON, no markdown fences."""


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
    max_iter = int(os.environ.get("MAX_ITER", 5))

    val = pd.read_parquet(P / "parquet/val.parquet")
    val_r1 = json.loads(EXP_PATH.read_text())
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

    print(f"[load] {LLM_MODEL}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    @torch.inference_mode()
    def llm_call(messages, max_new=3500):
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt", truncation=True, max_length=8192).to("cuda")
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=1.05,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def gen_expansions(system_prompt):
        results = {}
        for vq in val.itertuples():
            msgs = [
                {"role":"system","content":system_prompt},
                {"role":"user","content":f"English question:\n{vq.query}\n\nProduce the JSON."},
            ]
            text = llm_call(msgs, max_new=3500)
            parsed = parse_json_loose(text)
            results[vq.query_id] = {"raw": text, "parsed": parsed}
        return results

    def evaluate(expansions):
        rows = []
        for vq in val.itertuples():
            exp_r1 = val_r1.get(vq.query_id, {}).get("parsed") or {}
            mvp = expansions.get(vq.query_id, {}).get("parsed") or {}
            kws = exp_r1.get("keywords_de") or []
            codes = exp_r1.get("law_codes") or []
            d = {"qid": vq.query_id, "orig": vq.query,
                 "kw": " ".join(codes + kws) if (kws or codes) else None,
                 "hyde": exp_r1.get("hyde") if exp_r1.get("hyde") and len(exp_r1.get("hyde","")) >= 50 else None,
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
            rows.append(d)
        # warm
        for r in rows:
            for k in MODES:
                if r.get(k):
                    laws_for(r[k]); courts_for(r[k])
        preds = []
        for r in rows:
            rs_law, rs_court = [], []
            for k in MODES:
                v = r.get(k)
                if v:
                    rs_law.append(laws_for(v)); rs_court.append(courts_for(v))
            ls = rrf_merge(rs_law); cs = rrf_merge(rs_court)
            court_top = sorted(cs.items(), key=lambda x: -x[1])
            gf = Counter()
            for d, _ in court_top[:GRAPH_POOL]:
                for law in court_to_laws.get(d, []):
                    gf[law] += 1
            for law, freq in gf.items():
                ls[law] = ls.get(law, 0.0) + freq * GRAPH_W
            law_top = sorted(ls.items(), key=lambda x: -x[1])[:N_LAW]
            preds.append([d for d, _ in law_top])
        golds = [r["gold"] for r in rows]
        f1 = macro_f1(preds, golds)
        per_q = {qid: f1_per_query(p, g) for qid, p, g in zip([r["qid"] for r in rows], preds, golds)}
        return f1, per_q, preds, rows

    history = []
    if HIST_PATH.exists():
        history = json.loads(HIST_PATH.read_text())
        print(f"[resume] history: {len(history)} prior iterations")

    current_prompt = history[-1]["prompt"] if history else INITIAL_SYSTEM
    best_f1 = max((h["f1"] for h in history), default=-1.0)
    best_prompt = history[max(range(len(history)), key=lambda i: history[i]["f1"])]["prompt"] if history else current_prompt

    for it in range(len(history), len(history) + max_iter):
        print(f"\n{'='*70}")
        print(f"[iter {it}]")
        t0 = time.time()
        print("  [step 1] generating expansions with current prompt")
        exps = gen_expansions(current_prompt)
        n_ok = sum(1 for v in exps.values() if v.get("parsed"))
        print(f"  parsed {n_ok}/10 in {time.time()-t0:.1f}s")
        print("  [step 2] evaluating retrieval")
        t0 = time.time()
        f1, per_q, preds, rows = evaluate(exps)
        print(f"  F1 = {f1:.4f}  per_q={ {k: round(v,3) for k,v in per_q.items()} }  in {time.time()-t0:.1f}s")

        # Save snapshot
        snap = {
            "iter": it,
            "prompt": current_prompt,
            "f1": f1,
            "per_query_f1": per_q,
            "n_expansion_ok": n_ok,
            "ts": datetime.now().isoformat(),
        }
        history.append(snap)
        HIST_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2))

        if f1 > best_f1:
            print(f"  ★ new best F1 {f1:.4f} (prev {best_f1:.4f})")
            best_f1 = f1; best_prompt = current_prompt

        # Build critic input
        weak = sorted(per_q.items(), key=lambda x: x[1])[:3]
        feedback = []
        for qid, qf in weak:
            row = next(r for r in rows if r["qid"] == qid)
            pred = preds[[r["qid"] for r in rows].index(qid)]
            gold = row["gold"]
            missed = [g for g in gold if g not in set(pred)]
            wrong = [p for p in pred if p not in set(gold)]
            ex = exps.get(qid, {}).get("parsed") or {}
            feedback.append({
                "qid": qid, "f1": qf,
                "missed_examples": missed[:8],
                "wrong_examples": wrong[:5],
                "expansion_summary": {
                    "trans_concise": (ex.get("trans_concise") or "")[:150],
                    "hyde_statute": (ex.get("hyde_statute") or "")[:200],
                    "sub_questions": ex.get("sub_questions"),
                },
            })

        critic_user = (
            f"CURRENT SYSTEM PROMPT:\n```\n{current_prompt}\n```\n\n"
            f"VAL Macro F1 = {f1:.4f}\n"
            f"Per-query F1: {per_q}\n\n"
            f"WEAK QUERY ANALYSIS (3 weakest):\n"
            f"{json.dumps(feedback, ensure_ascii=False, indent=2)}\n\n"
            f"Now produce the JSON with rationale + revised_system_prompt."
        )
        print("  [step 3] critic LLM call")
        t0 = time.time()
        critic_text = llm_call([
            {"role":"system","content":CRITIC_SYSTEM},
            {"role":"user","content":critic_user},
        ], max_new=4000)
        critic_parsed = parse_json_loose(critic_text)
        print(f"  critic in {time.time()-t0:.1f}s, parsed={critic_parsed is not None}")

        if not critic_parsed or not critic_parsed.get("revised_system_prompt"):
            print("  ! critic failed, stopping")
            break

        new_prompt = critic_parsed["revised_system_prompt"]
        print(f"  rationale: {critic_parsed.get('rationale', '')[:200]}")
        if new_prompt == current_prompt:
            print("  critic returned identical prompt; stopping")
            break
        current_prompt = new_prompt
        snap["critic_rationale"] = critic_parsed.get("rationale", "")
        HIST_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2))

    print(f"\n[done] best F1 = {best_f1:.4f}")
    print(f"[best prompt saved at iter with score={best_f1:.4f}]")


if __name__ == "__main__":
    main()
