"""
Stage 8: regenerate expansions with thinking mode + stronger Swiss-law-specific
prompt. Targets:
  - Swiss-law-formatted article citations in HyDE (force "Art. N Abs. M XYZ").
  - Distinguish article number AND Absatz when relevant.
  - Encourage 5+ specific articles in HyDE.

Outputs:
  expansions/val_expansions_qwen3_14b_thinking.json
  expansions/test_expansions_qwen3_14b_thinking.json

Run with: python 08_expand_thinking.py [val|test]
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
BATCH = 1                       # thinking mode is heavier, smaller batch
MAX_NEW = 4096                  # thinking can be long
ENABLE_THINKING = True

EXPANSION_SYSTEM = """You are an expert in Swiss law analyzing English legal questions to retrieve Swiss legal sources.

Your job is to translate the English fact pattern into German legal search aids that match the precise citation form used in Swiss legal texts.

CITATION FORMAT (CRITICAL — match exactly): \"Art. N Abs. M lit. x XYZ\" where
  N = article number (digits, possibly with letter suffix like 3a, 11bis),
  M = Absatz (paragraph) number,
  lit. x = optional litera (a, b, c…),
  XYZ = abbreviation of the Swiss code (StPO, OR, ZGB, StGB, BV, BGG, IPRG, DBG, ZPO, AHVG, IVG, UVG, …).

RULES for the JSON you output:
  1. legal_areas: 1-4 Swiss legal areas in German (e.g. "Strafverfahrensrecht", "Obligationenrecht").
  2. law_codes: 2-6 abbreviation strings of Swiss codes most relevant to the question.
  3. keywords_de: 6-15 specific German legal terms / phrases likely to appear in the cited articles or court considerations. Avoid generic words.
  4. hyde: A 250-450 word German legal analysis. Cite AT LEAST 5 specific articles using the exact \"Art. N Abs. M XYZ\" form (include Absatz when relevant). Mention concrete legal doctrines, not abstract restatements of the question.

Output ONLY a single valid JSON object — no preamble, no trailing prose, no markdown fences.

Schema:
{
  \"legal_areas\": [...],
  \"law_codes\": [...],
  \"keywords_de\": [...],
  \"hyde\": \"...\"
}
"""


def build_prompt(tok, q, enable_thinking):
    msgs = [
        {"role": "system", "content": EXPANSION_SYSTEM},
        {"role": "user", "content": f"Legal question:\n\n{q}\n\nProduce the JSON now."},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(msgs, **kwargs, enable_thinking=enable_thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def parse_json_loose(text):
    # If thinking output is wrapped in <think>…</think>, pick text after the closing tag
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # First JSON object (greedy on outermost braces)
    try:
        # Find balanced braces using a simple scanner
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        return json.loads(text[start:end])
    except Exception:
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return None


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    assert which in ("val", "test")
    src = P / f"parquet/{which}.parquet"
    out_path = EXP_DIR / f"{which}_expansions_qwen3_14b_thinking.json"

    df = pd.read_parquet(src)
    print(f"[load] {which}: {len(df)} queries")

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
    print(f"[todo] {len(todo_ids)} queries (thinking={ENABLE_THINKING}, max_new={MAX_NEW})")

    print(f"[load] {LLM_MODEL}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    print(f"[load] model in {time.time() - t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    todo_df = df[df["query_id"].isin(todo_ids)].reset_index(drop=True)

    t_total = time.time()
    n_ok, n_fail = 0, 0
    for i, row in todo_df.iterrows():
        qid = row["query_id"]
        prompt = build_prompt(tok, row["query"], ENABLE_THINKING)
        inputs = tok([prompt], return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        t = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        elapsed = time.time() - t
        in_len = inputs["input_ids"].shape[1]
        text = tok.decode(out[0][in_len:], skip_special_tokens=True)
        parsed = parse_json_loose(text)
        cached[qid] = {"raw": text, "parsed": parsed}
        if parsed: n_ok += 1
        else: n_fail += 1
        n_new = out.shape[1] - in_len
        codes = (parsed or {}).get("law_codes", "?") if parsed else text[:80].replace("\n"," ")
        print(f"  [{qid}] {'OK' if parsed else 'FAIL'} ({n_new}t in {elapsed:.1f}s): {codes}")
        out_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2))

    print(f"[done] {which} total {time.time()-t_total:.1f}s | OK={n_ok} FAIL={n_fail}")


if __name__ == "__main__":
    main()
