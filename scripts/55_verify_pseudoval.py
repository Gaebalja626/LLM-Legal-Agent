"""
Stage 26: Plan-G — verification + missing-citation reasoning.

Pipeline:
  1. Round-1 retrieval with the stage-24 best config (val F1 0.1593):
       modes = orig, kw, hyde, hyde_statute, sub_q3, trans_concise
       → produce top-K_show (15 law + 15 court) candidates per query.
  2. Show LLM the English query + the candidates (citation + 220-char snippet)
     and ask for two lists in JSON:
       - "matched":    [citations from candidates that LLM judges relevant]
       - "additional": [Swiss-law citations NOT in the candidates that LLM
                       thinks should be in the answer; exact format:
                       'Art. N Abs. M XYZ' or 'BGE V S P E. N']
  3. Build several final-list variants and report Macro F1 on val:
       a) baseline (no LLM verify)
       b) baseline ∪ additional
       c) matched ∪ additional   (LLM-only)
       d) baseline ∩ matched ∪ additional   (filter baseline by matched, add new)
       e) (matched ∪ additional) padded by baseline to fixed size

Output: expansions/pseudo_val_verify_reasoning_qwen3_14b.json
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
EXP_PATH = P / "expansions/pseudo_val_expansions_qwen3_14b.json"
MV_PATH  = P / "expansions/pseudo_val_multiview_qwen3_14b.json"
CIT_PKL  = P / "parquet/citation_text_v2.pkl"
OUT_PATH = P / "expansions/pseudo_val_verify_reasoning_qwen3_14b.json"

LLM_MODEL = "Qwen/Qwen3-14B-AWQ"
K_LAW_RAW, K_COURT_RAW = 200, 200
GRAPH_POOL = 100
GRAPH_W = 0.05
SHOW_N_LAW = 15
SHOW_N_COURT = 15
SNIPPET_CHARS = 220
MAX_NEW = 1500
MODES_BEST = ("orig", "kw", "hyde", "hyde_statute", "sub_q3", "trans_concise")

VERIFY_SYSTEM = """You are a Swiss legal research assistant verifying a retrieval pipeline's output.

You will see:
  - An English Swiss-law question.
  - The top retrieved candidates: each is a citation + a German snippet from the actual Swiss legal corpus.

Your task:
  1. Identify which candidates are RELEVANT to answering the question (matched).
  2. Identify Swiss-law citations that should also be in the answer but are MISSING from the candidates (additional). Use exact citation form:
       Statutes: 'Art. N Abs. M lit. x XYZ'  (XYZ ∈ {StPO, StGB, OR, ZGB, BV, BGG, IPRG, DBG, ZPO, AHVG, IVG, UVG, BVG, SchKG, ATSG, …}).
       Cases:    'BGE V S P E. N' or docket forms like '5A_800/2019 E. 2'.

Output STRICTLY one JSON object:
{
  "matched":    [list of citation strings copied verbatim from the candidates that you judge relevant; subset of candidate citations],
  "additional": [list of NEW citations not in candidates but plausibly cited in the actual answer; max 25; precise format only]
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
    print("[load] data")
    test = pd.read_parquet(P / "parquet/pseudo_val.parquet")
    test_exp = json.loads(EXP_PATH.read_text())
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

    print("[load] BGE-M3")
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
        exp = test_exp.get(vq.query_id, {}).get("parsed") or {}
        mvp = mv.get(vq.query_id, {}).get("parsed") or {}
        kws = exp.get("keywords_de") or []
        codes = exp.get("law_codes") or []
        d = {"qid": vq.query_id, "orig": vq.query,
             "kw": " ".join(codes + kws) if (kws or codes) else None,
             "hyde": (exp.get("hyde") if exp.get("hyde") and len(exp.get("hyde","")) >= 50 else None),
             "trans_concise": mvp.get("trans_concise") or None,
             "hyde_statute":  mvp.get("hyde_statute")  or None,
             "gold": []}
        sq = mvp.get("sub_questions") or []
        d["sub_q3"] = sq[2] if len(sq) >= 3 and sq[2] and len(sq[2]) > 10 else None
        for k in list(d.keys()):
            if k in ("qid","orig","gold"): continue
            v = d.get(k)
            if not v or (isinstance(v, str) and len(v) < 20):
                d[k] = None
        return d

    rows = [build_row(vq) for vq in test.itertuples()]
    distinct = set()
    for r in rows:
        for k in MODES_BEST:
            if r.get(k): distinct.add(r[k])
    print(f"[search] pre-warm {len(distinct)} queries")
    t = time.time()
    for q in distinct:
        laws_for(q); courts_for(q)
    print(f"  done in {time.time()-t:.1f}s")

    def round1_lists(r):
        rs_law, rs_court = [], []
        for k in MODES_BEST:
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
        law_ranked = sorted(ls.items(), key=lambda x: -x[1])
        return law_ranked, court_top

    print("\n[round1] precomputing candidate pools")
    pools = {}
    for r in rows:
        law_r, court_r = round1_lists(r)
        pools[r["qid"]] = {
            "law":   [d for d, _ in law_r[:SHOW_N_LAW]],
            "court": [d for d, _ in court_r[:SHOW_N_COURT]],
            "law_ranked":   law_r,
            "court_ranked": court_r,
        }

    # ---------- LLM verify ----------
    cached = {}
    if OUT_PATH.exists():
        cached = json.loads(OUT_PATH.read_text())
    todo = [r for r in rows if cached.get(r["qid"], {}).get("parsed") is None]
    if todo:
        print(f"\n[load] {LLM_MODEL} (todo={len(todo)})")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, dtype=torch.float16, device_map="cuda")
        model.eval()
        print(f"  loaded in {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

        for r in todo:
            qid = r["qid"]
            cands = pools[qid]["law"] + pools[qid]["court"]
            display = []
            for c in cands:
                txt = (cit2text.get(c, "") or "")[:SNIPPET_CHARS].replace("\n", " ")
                display.append(f"  - {c}: {txt}")
            user_msg = (
                f"Question (English):\n{r['orig']}\n\n"
                f"Top retrieved candidates ({len(cands)}: {SHOW_N_LAW} laws + {SHOW_N_COURT} courts):\n"
                + "\n".join(display)
                + "\n\nProduce the JSON now."
            )
            msgs = [
                {"role": "system", "content": VERIFY_SYSTEM},
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
            cached[qid] = {"raw": text, "parsed": parsed}
            n_match = len((parsed or {}).get("matched", []) or [])
            n_add = len((parsed or {}).get("additional", []) or [])
            print(f"  [{qid}] {'OK' if parsed else 'FAIL'} ({time.time()-t:.1f}s): matched={n_match} additional={n_add}")
            OUT_PATH.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
        del model
        torch.cuda.empty_cache()

    # ---------- Build & evaluate variants ----------
    golds = [r["gold"] for r in rows]

    def baseline_pred(r, n_law=15, n_court=10):
        law_r = pools[r["qid"]]["law_ranked"]
        court_r = pools[r["qid"]]["court_ranked"]
        return [d for d, _ in law_r[:n_law]] + [d for d, _ in court_r[:n_court]]

    def variant_pred(r, kind, n_law=15, n_court=10, target_size=25):
        law_r = pools[r["qid"]]["law_ranked"]
        court_r = pools[r["qid"]]["court_ranked"]
        base = [d for d, _ in law_r[:n_law]] + [d for d, _ in court_r[:n_court]]
        verify = (cached.get(r["qid"]) or {}).get("parsed") or {}
        matched = list(verify.get("matched", []) or [])
        additional = list(verify.get("additional", []) or [])
        if kind == "baseline":
            return base
        if kind == "base+add":
            seen = set(); out = []
            for c in base + additional:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]
        if kind == "matched+add":
            seen = set(); out = []
            for c in matched + additional:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]
        if kind == "base_filter_matched_plus_add":
            keep = [c for c in base if c in set(matched)]
            seen = set(keep); out = list(keep)
            for c in additional:
                if c not in seen:
                    seen.add(c); out.append(c)
            # pad with rest of base if short
            for c in base:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]
        if kind == "ma_pad_base":
            seen = set(); out = []
            for c in matched + additional:
                if c not in seen:
                    seen.add(c); out.append(c)
            for c in base:
                if c not in seen:
                    seen.add(c); out.append(c)
            return out[:target_size]
        raise ValueError(kind)

    print("\n[eval] verify variants (target_size=25)")
    for kind in ["baseline", "base+add", "matched+add",
                 "base_filter_matched_plus_add", "ma_pad_base"]:
        preds = [variant_pred(r, kind) for r in rows]
        f1 = macro_f1(preds, golds)
        per_q = [round(f1_per_query(p, g), 3) for p, g in zip(preds, golds)]
        print(f"  {kind:<32}: {f1:.4f}  {per_q}")

    print("\n[eval] base+add at different output sizes")
    for sz in [20, 22, 25, 30, 35, 40]:
        preds = [variant_pred(r, "base+add", target_size=sz) for r in rows]
        f1 = macro_f1(preds, golds)
        print(f"  base+add target_size={sz}: {f1:.4f}")


if __name__ == "__main__":
    main()
