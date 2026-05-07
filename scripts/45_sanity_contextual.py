"""
Sanity: contextual retrieval augmentation on BGE subset.

Pool: random 1000 BGE + all val gold BGE (~64) → ~1064 unique BGE.
For each, generate a short bilingual contextual label via Qwen3-14B-AWQ.
Build two FAISS indexes:
  - raw text only (baseline)
  - context_en + context_de + text (augmented)

Then for each val query (using best multi-view variants), retrieve top-30 from
both indexes and measure how many gold BGE appear.
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, pickle, time, re
from pathlib import Path
from collections import Counter, defaultdict

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
OUT_LABELS = P / "expansions/sanity_contextual_labels.json"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
N_RANDOM = 1000
BATCH = 4
MAX_NEW = 350
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")

CTX_SYSTEM = """You see an excerpt from a Swiss Federal Tribunal decision (BGE). Write a brief contextual summary in BOTH English and German describing what specific legal issue, statute, and situation this excerpt addresses. 1-2 sentences each.

Output STRICTLY one JSON:
{
  "context_de": "<1-2 sentence German summary>",
  "context_en": "<1-2 sentence English summary>"
}

Output ONLY the JSON."""


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]

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
    with open(P / "indexes/bge_only_doc_ids_v2.pkl", "rb") as f:
        BGE_DOC_IDS = pickle.load(f)
    BGE_VOCAB = set(BGE_DOC_IDS)
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)

    # Pool: random + val gold BGEs
    rng = np.random.default_rng(42)
    rand_idx = rng.choice(len(BGE_DOC_IDS), size=N_RANDOM, replace=False)
    pool = [BGE_DOC_IDS[int(i)] for i in rand_idx]
    gold_bges = []
    for vq in val.itertuples():
        for g in split_cites(vq.gold_citations):
            if g in BGE_VOCAB:
                gold_bges.append(g)
    gold_bges = list(dict.fromkeys(gold_bges))
    print(f"  random pool: {len(pool)}, val gold BGE: {len(gold_bges)}")
    pool_set = set(pool); pool_set |= set(gold_bges)
    pool = list(pool_set)
    print(f"  total pool: {len(pool)}")

    # Generate contextual labels (cached per-citation)
    cached = {}
    if OUT_LABELS.exists():
        cached = json.loads(OUT_LABELS.read_text())
    todo = [c for c in pool if c not in cached or cached[c].get("parsed") is None]
    print(f"  todo for label generation: {len(todo)}")

    if todo:
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

        t_total = time.time()
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i+BATCH]
            prompts = []
            for c in batch:
                txt = (cit2text.get(c, "") or "")[:1500]
                msgs = [
                    {"role":"system","content": CTX_SYSTEM},
                    {"role":"user","content": f"Citation: {c}\n\nExcerpt:\n{txt}\n\nProduce the JSON."},
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
            for j, c in enumerate(batch):
                text = tok.decode(out[j][in_len:], skip_special_tokens=True)
                parsed = parse_json_loose(text)
                cached[c] = {"raw": text, "parsed": parsed}
            elapsed = time.time() - t
            done = i + len(batch)
            eta = (len(todo) - done) / max(1, done) * (time.time() - t_total)
            n_ok = sum(1 for c in batch if cached[c].get("parsed"))
            print(f"  [{done}/{len(todo)}] batch {elapsed:.1f}s, ok={n_ok}/{len(batch)}, eta={eta:.0f}s")
            OUT_LABELS.write_text(json.dumps(cached, ensure_ascii=False))
        del model; torch.cuda.empty_cache()
        print(f"[done] generation total {time.time()-t_total:.1f}s")
    n_ok = sum(1 for c in pool if cached.get(c, {}).get("parsed"))
    print(f"  parsed labels: {n_ok}/{len(pool)}")

    # Show 3 example labels
    print("\n[3 example contextual labels]")
    for c in pool[:3]:
        p = cached.get(c, {}).get("parsed")
        if not p: continue
        print(f"\n  {c!r}")
        print(f"    raw text[:150]: {(cit2text.get(c) or '')[:150]}...")
        print(f"    context_de: {p.get('context_de', '')[:200]}")
        print(f"    context_en: {p.get('context_en', '')[:200]}")

    # Build two BGE-pool embeddings
    print("\n[load] BGE-M3 encoder")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    raw_texts = []
    aug_texts = []
    pool_keep = []  # only those with successful labels
    for c in pool:
        p = cached.get(c, {}).get("parsed")
        if not p: continue
        ctx_de = p.get("context_de", "") or ""
        ctx_en = p.get("context_en", "") or ""
        raw = cit2text.get(c, "") or ""
        raw_texts.append(raw)
        aug_texts.append(f"{ctx_en} {ctx_de} {raw}".strip())
        pool_keep.append(c)
    print(f"  pool with labels: {len(pool_keep)}")

    print("[encode] raw texts (1)")
    raw_emb = enc.encode(raw_texts, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64).astype("float32")
    print("[encode] augmented texts (2)")
    aug_emb = enc.encode(aug_texts, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64).astype("float32")

    raw_idx = faiss.IndexFlatIP(raw_emb.shape[1]); raw_idx.add(raw_emb)
    aug_idx = faiss.IndexFlatIP(aug_emb.shape[1]); aug_idx.add(aug_emb)

    def encode_q(q):
        return enc.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype("float32")

    print("\n[eval] for each val query, top-30 in raw vs aug")

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
    pool_keep_set = set(pool_keep)

    raw_total_hit = 0
    aug_total_hit = 0
    raw_total_in_top10 = 0
    aug_total_in_top10 = 0
    total_gold_in_pool = 0

    for r in rows:
        gold_bge = [g for g in r["gold"] if g in BGE_VOCAB and g in pool_keep_set]
        if not gold_bge: continue
        total_gold_in_pool += len(gold_bge)

        # encode all 6 variants and union top-30 from each index
        raw_union = set(); aug_union = set()
        raw_top10_union = set(); aug_top10_union = set()
        for k in MODES:
            v = r.get(k)
            if not v: continue
            qv = encode_q(v)
            sr, ir = raw_idx.search(qv, 30)
            sa, ia = aug_idx.search(qv, 30)
            raw_top = [pool_keep[idx] for idx in ir[0]]
            aug_top = [pool_keep[idx] for idx in ia[0]]
            raw_union |= set(raw_top); aug_union |= set(aug_top)
            raw_top10_union |= set(raw_top[:10])
            aug_top10_union |= set(aug_top[:10])

        rh = sum(1 for g in gold_bge if g in raw_union)
        ah = sum(1 for g in gold_bge if g in aug_union)
        rh10 = sum(1 for g in gold_bge if g in raw_top10_union)
        ah10 = sum(1 for g in gold_bge if g in aug_top10_union)
        raw_total_hit += rh; aug_total_hit += ah
        raw_total_in_top10 += rh10; aug_total_in_top10 += ah10
        print(f"  {r['qid']}: gold-in-pool={len(gold_bge)}  raw_top30_hit={rh}  aug_top30_hit={ah}   raw_top10={rh10}  aug_top10={ah10}")

    print(f"\n[summary]")
    print(f"  total gold BGE in pool: {total_gold_in_pool}")
    print(f"  raw  top-30 union recall: {raw_total_hit}/{total_gold_in_pool} = {raw_total_hit/max(1,total_gold_in_pool):.3f}")
    print(f"  aug  top-30 union recall: {aug_total_hit}/{total_gold_in_pool} = {aug_total_hit/max(1,total_gold_in_pool):.3f}")
    print(f"  raw  top-10 union recall: {raw_total_in_top10}/{total_gold_in_pool} = {raw_total_in_top10/max(1,total_gold_in_pool):.3f}")
    print(f"  aug  top-10 union recall: {aug_total_in_top10}/{total_gold_in_pool} = {aug_total_in_top10/max(1,total_gold_in_pool):.3f}")


if __name__ == "__main__":
    main()
