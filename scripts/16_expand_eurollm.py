"""
Stage 16: regenerate val (and optionally test) expansions using EuroLLM-9B-Instruct.

Hypothesis: Qwen3-14B weak in German legal vocabulary; EuroLLM is EU-language native.

Same JSON schema as 03_test_expansion.py / 14_round2_gen.py for downstream compat.

Usage:
  python 16_expand_eurollm.py val
  python 16_expand_eurollm.py test
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

LLM_MODEL = "utter-project/EuroLLM-9B-Instruct"
BATCH = 2
MAX_NEW = 1500

EXPANSION_SYSTEM = """You are an expert in Swiss law. You will analyze a legal question (in English, often a long fact pattern) and produce structured search aids in German.

Your task: identify legal issues, extract precise German legal keywords (real terms used in Swiss codes), and write a hypothetical German legal analysis.

CITATION FORMAT used by Swiss texts: \"Art. N Abs. M lit. x XYZ\" where XYZ is the code abbreviation (StPO, OR, ZGB, StGB, BV, BGG, IPRG, DBG, ZPO, AHVG, IVG, UVG, BVG, SchKG, ATSG, …).

Output STRICTLY valid JSON:
{
  \"legal_areas\": [list of relevant Swiss legal areas in German],
  \"law_codes\": [list of Swiss law abbreviations likely relevant, 2-8 items],
  \"keywords_de\": [precise German legal terms / phrases, 6-15 items, no English, no duplicates],
  \"hyde\": \"A 250-450 word German legal analysis using precise legal terminology. Cite specific articles in the form 'Art. N Abs. M XYZ' when natural.\"
}

Output ONLY the JSON, no markdown fences, no preamble."""


def build_prompt(tok, q):
    msgs = [
        {"role": "system", "content": EXPANSION_SYSTEM},
        {"role": "user",   "content": f"Legal question:\n\n{q}\n\nProduce the JSON now."},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_json_loose(text):
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
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    assert which in ("val", "test")
    src = P / f"parquet/{which}.parquet"
    out_path = EXP_DIR / f"{which}_expansions_eurollm9b.json"

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
    print(f"[todo] {len(todo_ids)} queries (max_new={MAX_NEW}, batch={BATCH})")

    print(f"[load] {LLM_MODEL}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"[load] model in {time.time() - t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    todo_df = df[df["query_id"].isin(todo_ids)].reset_index(drop=True)
    prompts = [build_prompt(tok, q) for q in todo_df["query"].tolist()]

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
                repetition_penalty=1.05,
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
            preview = (parsed or {}).get("law_codes", "?") if parsed else text[:80].replace("\n"," ")
            print(f"  [{qid}] {tag}: {preview}")
        print(f"  batch {i//BATCH+1}/{(len(prompts)+BATCH-1)//BATCH}: {elapsed:.1f}s")
        out_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2))

    print(f"[done] {which} total {time.time()-t_total:.1f}s | OK={n_ok} FAIL={n_fail}")


if __name__ == "__main__":
    main()
