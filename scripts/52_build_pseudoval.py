"""
Build pseudo-val by translating 100 random train queries to English.

Sample 100 queries from train.parquet, use Qwen3-14B-AWQ to translate
German → English (preserving facts/citations/dates), and save as
parquet/pseudo_val.parquet with the SAME schema as val.parquet:
  - query_id  ('pseudo_001'..'pseudo_100')
  - query     (English translation)
  - gold_citations (original)
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

P = Path("~/legal-ir").expanduser()
OUT_PARQ = P / "parquet/pseudo_val.parquet"
OUT_RAW = P / "parquet/pseudo_val_raw.json"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
N_SAMPLE = 100
SEED = 42
BATCH = 4
MAX_NEW = 1500

SYSTEM = (
    "You translate Swiss-law German questions into natural fluent English questions. "
    "Preserve every fact, date, citation, person name, statute reference, party identifiers, and number EXACTLY. "
    "Do not summarize or shorten. Output ONLY the English translation — no preamble, no explanation."
)


def main():
    if OUT_PARQ.exists():
        print(f"[skip] {OUT_PARQ} exists")
        return
    train = pd.read_parquet(P / "parquet/train.parquet")
    print(f"[load] train: {len(train):,}")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train), size=N_SAMPLE, replace=False)
    sample = train.iloc[sorted(idx)].reset_index(drop=True)
    print(f"  sampled: {len(sample)}")

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

    translations = []
    raw = {}
    t_total = time.time()
    for i in range(0, len(sample), BATCH):
        batch = sample.iloc[i:i+BATCH]
        prompts = []
        for _, row in batch.iterrows():
            msgs = [
                {"role":"system","content":SYSTEM},
                {"role":"user","content":row["query"]},
            ]
            try:
                p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=4096).to("cuda")
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 repetition_penalty=1.05,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        in_len = inputs["input_ids"].shape[1]
        for j, (_, row) in enumerate(batch.iterrows()):
            text = tok.decode(out[j][in_len:], skip_special_tokens=True).strip()
            translations.append(text)
            raw[row["query_id"]] = text
        done = i + len(batch)
        eta = (len(sample) - done) / max(1, done) * (time.time() - t_total)
        print(f"  [{done}/{len(sample)}] batch {time.time()-t:.1f}s, eta={eta:.0f}s")

    pseudo_val = pd.DataFrame({
        "query_id": [f"pseudo_{i+1:03d}" for i in range(len(sample))],
        "query": translations,
        "gold_citations": sample["gold_citations"].tolist(),
        "orig_query_id": sample["query_id"].tolist(),
    })
    pseudo_val.to_parquet(OUT_PARQ, compression="zstd")
    OUT_RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT_PARQ} ({len(pseudo_val)} rows, {OUT_PARQ.stat().st_size/1e3:.0f} KB)")
    print(f"[saved] {OUT_RAW}")
    print(f"[done] total {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
