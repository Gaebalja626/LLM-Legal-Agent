"""
Sanity: doctrinal-focused augmentation A2.

Generate per-BGE:
  - rule_de: one-sentence abstract legal rule (German)
  - rule_en: one-sentence English rule
  - cited_articles: list of cited article citations
  - key_terms_de: list of key German legal terms

Compare 3 retrieval modes:
  raw        — text only
  A1 (old)   — context_de + context_en + raw  (from sanity_contextual_labels)
  A2 (new)   — rule_de + rule_en + cited_articles + key_terms_de + raw

Per query × 6 multi-view variants, measure top-K recall on gold-in-pool.
"""
import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')

import json, pickle, time, re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

P = Path("~/legal-ir").expanduser()
EXP_PATH = P / "expansions/val_expansions_qwen3_14b.json"
MV_PATH  = P / "expansions/val_multiview_qwen3_14b.json"
A1_LABELS_PATH = P / "expansions/sanity_contextual_labels.json"
A2_LABELS_PATH = P / "expansions/sanity_doctrinal_labels.json"
CIT_PKL  = P / "parquet/citation_text_v2.pkl"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
BATCH = 4
MAX_NEW = 400
MODES = ("orig","kw","hyde","hyde_statute","sub_q3","trans_concise")

A2_SYSTEM = """You see an excerpt from a Swiss Federal Tribunal decision (BGE). Extract:
  1. The ABSTRACT LEGAL RULE this excerpt formulates (one sentence each in German and English). NOT case-specific facts — only the doctrinal rule, principle, or test.
  2. The articles/statutes CITED in the excerpt (precise format: "Art. N Abs. M XYZ"; XYZ ∈ StPO/StGB/OR/ZGB/BV/BGG/IPRG/DBG/ZPO/AHVG/IVG/UVG/BVG/SchKG/EMRK/Cst/CPP/CO/CC etc.).
  3. KEY GERMAN LEGAL TERMS used in the excerpt (5-10 specific terms — Untersuchungshaft, Verhältnismässigkeit, Kollusionsgefahr, etc.).

Output STRICTLY one JSON object:
{
  "rule_de": "<one sentence abstract legal rule in German>",
  "rule_en": "<one sentence abstract legal rule in English>",
  "cited_articles": [<list of article strings>],
  "key_terms_de": [<list of key German legal terms>]
}

Output ONLY the JSON, nothing else."""


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


def split_cites(s):
    if pd.isna(s) or s == "": return []
    return [x.strip() for x in s.split(";") if x.strip()]


def main():
    val = pd.read_parquet(P / "parquet/val.parquet")
    val_exp = json.loads(EXP_PATH.read_text())
    mv = json.loads(MV_PATH.read_text())
    a1_labels = json.loads(A1_LABELS_PATH.read_text())
    with open(P / "indexes/bge_only_doc_ids_v2.pkl", "rb") as f:
        BGE_DOC_IDS = pickle.load(f)
    BGE_VOCAB = set(BGE_DOC_IDS)
    with open(CIT_PKL, "rb") as f:
        cit2text = pickle.load(f)

    # use the same pool as A1 sanity (1067 BGEs with successful labels)
    pool = [c for c, lab in a1_labels.items() if lab.get("parsed")]
    print(f"[pool] {len(pool)} BGEs (same as A1 sanity)")

    # Generate A2 labels
    a2_labels = {}
    if A2_LABELS_PATH.exists():
        a2_labels = json.loads(A2_LABELS_PATH.read_text())
    todo = [c for c in pool if c not in a2_labels or a2_labels[c].get("parsed") is None]
    print(f"  todo for A2 generation: {len(todo)}")

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
                    {"role":"system","content":A2_SYSTEM},
                    {"role":"user","content":f"Citation: {c}\n\nExcerpt:\n{txt}\n\nProduce the JSON."},
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
                a2_labels[c] = {"raw": text, "parsed": parsed}
            done = i + len(batch)
            elapsed = time.time() - t
            n_ok = sum(1 for c in batch if a2_labels[c].get("parsed"))
            eta = (len(todo) - done) / max(1, done) * (time.time() - t_total)
            print(f"  [{done}/{len(todo)}] batch {elapsed:.1f}s, ok={n_ok}/{len(batch)}, eta={eta:.0f}s")
            A2_LABELS_PATH.write_text(json.dumps(a2_labels, ensure_ascii=False))
        del model; torch.cuda.empty_cache()
        print(f"[done] A2 generation total {time.time()-t_total:.1f}s")
    n_ok = sum(1 for c in pool if a2_labels.get(c, {}).get("parsed"))
    print(f"  A2 parsed: {n_ok}/{len(pool)}")

    # Show 3 examples
    print("\n[3 A2 example labels]")
    for c in pool[:3]:
        p = a2_labels.get(c, {}).get("parsed")
        if not p: continue
        print(f"\n  {c!r}")
        print(f"    rule_de: {p.get('rule_de', '')[:200]}")
        print(f"    rule_en: {p.get('rule_en', '')[:200]}")
        print(f"    cited:   {p.get('cited_articles', [])}")
        print(f"    terms:   {p.get('key_terms_de', [])}")

    # Build pool keep + 3 text variants per BGE
    print("\n[load] BGE-M3 encoder")
    enc = SentenceTransformer("BAAI/bge-m3", device="cuda")
    enc.max_seq_length = 256

    pool_keep = []
    raw_texts = []
    a1_texts = []
    a2_texts = []
    for c in pool:
        a1 = a1_labels.get(c, {}).get("parsed")
        a2 = a2_labels.get(c, {}).get("parsed")
        if not a1 or not a2: continue
        raw = cit2text.get(c, "") or ""
        if not raw: continue
        pool_keep.append(c)
        raw_texts.append(raw)
        a1_texts.append(f"{a1.get('context_en','')} {a1.get('context_de','')} {raw}".strip())
        rule_de = a2.get("rule_de") or ""
        rule_en = a2.get("rule_en") or ""
        cited = " ".join(a2.get("cited_articles") or [])
        terms = " ".join(a2.get("key_terms_de") or [])
        a2_texts.append(f"{rule_en} {rule_de} {cited} {terms} {raw}".strip())
    print(f"  pool with both labels: {len(pool_keep)}")

    print("[encode] raw, a1, a2")
    raw_emb = enc.encode(raw_texts, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=64).astype("float32")
    a1_emb = enc.encode(a1_texts, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=64).astype("float32")
    a2_emb = enc.encode(a2_texts, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=64).astype("float32")
    raw_idx = faiss.IndexFlatIP(raw_emb.shape[1]); raw_idx.add(raw_emb)
    a1_idx  = faiss.IndexFlatIP(a1_emb.shape[1]);  a1_idx.add(a1_emb)
    a2_idx  = faiss.IndexFlatIP(a2_emb.shape[1]);  a2_idx.add(a2_emb)

    def encode_q(q):
        return enc.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype("float32")

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
    pool_set = set(pool_keep)

    def search(idx, qv, k):
        s, i = idx.search(qv, k)
        return [pool_keep[idx_] for idx_ in i[0]]

    print("\n[eval] per-query top-K recall (gold BGE in pool)")
    for tag, idx in [("raw", raw_idx), ("A1 (general context)", a1_idx), ("A2 (doctrinal)", a2_idx)]:
        for K in [3, 5, 10, 20, 30]:
            total_hit, total_gold = 0, 0
            for r in rows:
                gold = [g for g in r["gold"] if g in BGE_VOCAB and g in pool_set]
                if not gold: continue
                total_gold += len(gold)
                union = set()
                for k_mode in MODES:
                    v = r.get(k_mode)
                    if v:
                        union |= set(search(idx, encode_q(v), K))
                total_hit += sum(1 for g in gold if g in union)
            print(f"  {tag:<25} K={K:>3} union over 6 views: hits {total_hit}/{total_gold} = {total_hit/max(1,total_gold):.3f}")
        print()

    # Per-view comparison (single-view, not union) for one query (val_001)
    print("\n[per-view single-view recall on val_001 (gold BGE in pool)]")
    r1 = next(r for r in rows if r["qid"] == "val_001")
    gold_in_pool = [g for g in r1["gold"] if g in BGE_VOCAB and g in pool_set]
    print(f"  val_001 gold in pool: {len(gold_in_pool)}")
    for view in MODES:
        v = r1.get(view)
        if not v: continue
        qv = encode_q(v)
        for tag, idx in [("raw", raw_idx), ("A1", a1_idx), ("A2", a2_idx)]:
            top10 = set(search(idx, qv, 10))
            top30 = set(search(idx, qv, 30))
            h10 = sum(1 for g in gold_in_pool if g in top10)
            h30 = sum(1 for g in gold_in_pool if g in top30)
            print(f"  view={view:<14} {tag:<3}: top10={h10}/{len(gold_in_pool)}  top30={h30}/{len(gold_in_pool)}")
        print()


if __name__ == "__main__":
    main()
