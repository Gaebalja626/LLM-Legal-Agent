"""
Stage 3: produce LLM expansions for test (40 queries) — same prompt/format as val.
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
OUT_PATH = EXP_DIR / "test_expansions_qwen3_14b.json"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
BATCH = 2
MAX_NEW = 1500

EXPANSION_SYSTEM = """You are an expert in Swiss law. You will analyze a legal question (in English, often a long fact pattern) and produce structured search aids in German.

Your task: identify legal issues, extract German legal keywords, and write a hypothetical German legal analysis.

Output STRICTLY valid JSON with these keys:
{
  "legal_areas": [list of relevant Swiss legal areas, e.g. "Strafverfahrensrecht", "Obligationenrecht"],
  "law_codes": [list of Swiss law abbreviations likely relevant, e.g. "StPO", "OR", "ZGB", "StGB", "BV", "BGG", "IPRG", "DBG"],
  "keywords_de": [German legal keywords/phrases, 5-15 items, specific legal terms not generic words],
  "hyde": "A hypothetical 200-400 word German legal analysis answering the question, using precise legal terminology. Cite article numbers when natural. Write as if for a legal opinion."
}

Output ONLY the JSON, no other text."""


def build_prompt(tok, q):
    msgs = [
        {"role": "system", "content": EXPANSION_SYSTEM},
        {"role": "user", "content": f"Legal question:\n\n{q}\n\nProduce the JSON now."},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(msgs, **kwargs, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def parse_json_loose(text):
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None


def main():
    test = pd.read_parquet(P / "parquet/test.parquet")
    print(f"[load] test: {len(test)} queries")

    cached = {}
    if OUT_PATH.exists():
        cached = json.loads(OUT_PATH.read_text())
    todo_ids = [
        qid for qid in test["query_id"].tolist()
        if cached.get(qid, {}).get("parsed") is None
    ]
    if not todo_ids:
        print("[done] all cached")
        return

    print(f"[todo] {len(todo_ids)} queries")
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

    todo_df = test[test["query_id"].isin(todo_ids)].reset_index(drop=True)
    prompts = [build_prompt(tok, q) for q in todo_df["query"].tolist()]
    print(f"[gen] batch={BATCH}, max_new={MAX_NEW}")

    t_total = time.time()
    n_ok, n_fail = 0, 0
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i:i + BATCH]
        qids = todo_df["query_id"].iloc[i:i + BATCH].tolist()
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=4096).to("cuda")
        t = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        elapsed = time.time() - t
        in_len = inputs["input_ids"].shape[1]
        for j, qid in enumerate(qids):
            text = tok.decode(out[j][in_len:], skip_special_tokens=True)
            parsed = parse_json_loose(text)
            cached[qid] = {"raw": text, "parsed": parsed}
            tag = "OK" if parsed else "FAIL"
            if parsed: n_ok += 1
            else: n_fail += 1
            preview = (parsed or {}).get("law_codes", "?") if parsed else text[:60].replace("\n"," ")
            print(f"  [{qid}] {tag}: {preview}")
        print(f"  batch {i//BATCH+1}/{(len(prompts)+BATCH-1)//BATCH}: {elapsed:.1f}s")
        OUT_PATH.write_text(json.dumps(cached, ensure_ascii=False, indent=2))

    print(f"[done] total {time.time() - t_total:.1f}s | parsed: {n_ok} OK, {n_fail} FAIL")
    OUT_PATH.write_text(json.dumps(cached, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
