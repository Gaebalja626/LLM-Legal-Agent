"""
Stage 20: ask LLM to translate the English query into a natural German question.

Output: expansions/{which}_translation_qwen3_14b.json — dict[qid] = {
  'raw': str, 'translation_de': str
}

Different from hyde: hyde is a German statement (hypothetical answer, ~300 words);
this is a faithful German rendering of the question itself, much shorter.
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, time, sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

P = Path("~/legal-ir").expanduser()
EXP_DIR = P / "expansions"
EXP_DIR.mkdir(exist_ok=True)

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
BATCH = 4
MAX_NEW = 700

SYSTEM = (
    "You translate English Swiss-law questions into natural, fluent German questions. "
    "Use precise Swiss German legal terminology. Preserve every fact, citation, name, date, and number exactly. "
    "Output ONLY the German translation — no preamble, no explanation, no markdown."
)


def build_prompt(tok, q):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": q},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    assert which in ("val", "test")
    df = pd.read_parquet(P / f"parquet/{which}.parquet")
    out_path = EXP_DIR / f"{which}_translation_qwen3_14b.json"

    cached = {}
    if out_path.exists():
        cached = json.loads(out_path.read_text())
    todo_ids = [
        qid for qid in df["query_id"].tolist()
        if not cached.get(qid, {}).get("translation_de")
    ]
    if not todo_ids:
        print("[done] all cached")
        return
    print(f"[load] {which}: {len(df)} | todo: {len(todo_ids)}")

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
    prompts = [build_prompt(tok, q) for q in todo_df["query"].tolist()]

    t_total = time.time()
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i:i+BATCH]
        qids = todo_df["query_id"].iloc[i:i+BATCH].tolist()
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=4096).to("cuda")
        t = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW, do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        for j, qid in enumerate(qids):
            text = tok.decode(out[j][in_len:], skip_special_tokens=True).strip()
            cached[qid] = {"raw": text, "translation_de": text}
            preview = text[:120].replace("\n", " ")
            print(f"  [{qid}] ({len(text)} chars): {preview}...")
        print(f"  batch {i//BATCH+1}/{(len(prompts)+BATCH-1)//BATCH}: {time.time()-t:.1f}s")
        out_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
    print(f"[done] {which} total {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
