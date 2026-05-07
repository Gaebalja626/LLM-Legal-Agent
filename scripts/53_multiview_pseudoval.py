"""
Stage 22: multi-view expansion — single LLM call producing 6 different German
views of the same English query.

Output: expansions/{which}_multiview_qwen3_14b.json — dict[qid] = {
  'raw': str, 'parsed': {
    'trans_natural':  '...',
    'trans_formal':   '...',
    'trans_concise':  '...',
    'hyde_statute':   '...',
    'hyde_case':      '...',
    'sub_questions':  ['...', '...', '...'],
  }
}

Usage: python 22_multiview_expand.py [val|test]
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, time, re, sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

P = Path("~/legal-ir").expanduser()
EXP_DIR = P / "expansions"
EXP_DIR.mkdir(exist_ok=True)

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
BATCH = 1                            # multi-view long output -> single
MAX_NEW = 3500

SYSTEM = """You are a Swiss legal research assistant. Given an English Swiss-law question, produce SIX different German views useful for retrieval over a German legal corpus.

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


def build_prompt(tok, q):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"English question:\n{q}\n\nProduce the JSON."},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_json_loose(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    try:
        start = text.find("{")
        if start == -1: return None
        depth = 0; end = -1; in_str = False; esc = False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False; continue
            if ch == "\\":
                esc = True; continue
            if ch == '"':
                in_str = not in_str; continue
            if in_str:
                continue
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
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    assert which in ("val", "test", "pseudo_val")
    df = pd.read_parquet(P / f"parquet/{which}.parquet")
    out_path = EXP_DIR / f"{which}_multiview_qwen3_14b.json"

    cached = {}
    if out_path.exists():
        cached = json.loads(out_path.read_text())
    todo_ids = [
        qid for qid in df["query_id"].tolist()
        if cached.get(qid, {}).get("parsed") is None
    ]
    if not todo_ids:
        print("[done] all cached")
        return
    print(f"[load] {which}: {len(df)} | todo: {len(todo_ids)} | max_new={MAX_NEW}")

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

    todo_df = df[df["query_id"].isin(todo_ids)].reset_index(drop=True)

    t_total = time.time()
    n_ok, n_fail = 0, 0
    for _, row in todo_df.iterrows():
        qid = row["query_id"]
        prompt = build_prompt(tok, row["query"])
        inputs = tok([prompt], return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        t = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW, do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_json_loose(text)
        cached[qid] = {"raw": text, "parsed": parsed}
        if parsed: n_ok += 1
        else: n_fail += 1
        keys = list(parsed.keys()) if parsed else "FAIL"
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"  [{qid}] {'OK' if parsed else 'FAIL'} ({n_new}t in {time.time()-t:.1f}s): keys={keys}")
        out_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2))

    print(f"[done] {which} total {time.time()-t_total:.1f}s | OK={n_ok} FAIL={n_fail}")


if __name__ == "__main__":
    main()
