# LLM Legal Agent — Swiss Legal Information Retrieval

Kaggle "LLM Agentic Legal Information Retrieval" (Omnilex).
영어 법률 질문 40개 → 스위스 법률 코퍼스에서 인용(citation) retrieval. Macro F1 평가.

## 🏆 결과

| Submission | val F1 | **public LB** |
|---|---|---|
| sub_baseline_hybrid (시작점) | 0.000 | 0.000 |
| sub_v2_clean (LLM 없이, BGE-M3 dense) | 0.0035 | 0.00679 |
| sub_v4_oL15_oC10 (round-1 expansion baseline) | 0.1447 | 0.11455 |
| sub_v6_multiview_oC0 (multi-view + nC=0) | 0.2064 | 0.14668 |
| **sub_v7_planG_oL15_oC0_t17** | **0.2088** | **0.15231** ⭐ |
| sub_v8_combined (+graph C, val ↑ but LB ↓ over-fit) | 0.2214 | 0.15192 |

**Baseline 0.11455 → 0.15231 = +0.038 absolute, +33% relative.**
LLM-free 베이스라인 대비 **22배** 향상.

## 🏗️ Best 파이프라인 (sub_v7)

```
INPUT: English Swiss-law narrative (~1500자)
    ↓
[STAGE 1] LLM Round-1 Expansion (Qwen3-14B-AWQ via gptqmodel/Marlin)
    Output JSON: {legal_areas, law_codes, keywords_de, hyde}
    ↓
[STAGE 2] LLM Multi-view Expansion (별도 호출)
    Output JSON: {trans_natural, trans_formal, trans_concise,
                  hyde_statute, hyde_case, sub_questions[3]}
    ↓
[STAGE 3] Build 6 query variants
    orig (English original)
    kw   (codes + keywords joined)
    hyde (round-1 German narrative)
    hyde_statute (multi-view, 조문 스타일)
    sub_q3 (multi-view, sub-question 3)
    trans_concise (multi-view, 압축 번역)
    ↓
[STAGE 4] Split Dense Retrieval (BGE-M3, max_seq=256, normalize)
    for each q in 6 variants:
      law_top200   = law_index.search(q, 200)
      court_top200 = court_index.search(q, 200)
    ↓
[STAGE 5] RRF Fusion (per source, k=60)
    law_scores[d]   = Σ 1/(60 + rank_q(d) + 1)
    court_scores[d] = Σ 1/(60 + rank_q(d) + 1)
    ↓
[STAGE 6] Court→Law Graph 1-hop Boost
    graph_freq[law] = Σ_{c ∈ court_top100} 1[law ∈ court_to_laws[c]]
    law_scores[law] += graph_freq[law] * 0.05
    ↓
[STAGE 7] LLM Verify Reasoning (Plan G)
    Top-30 candidates (15 law + 15 court) → LLM verify call:
      output: matched + additional citations
    ↓
[STAGE 8] Output
    base = top-15 law (no court output, nC=0)
    final = dedup(base + plan_g_additional)[:17]
    ↓
OUTPUT CSV: query_id, predicted_citations (semicolon-joined)
```

## 🔑 핵심 하이퍼파라미터 (best)

| 파라미터 | 값 |
|---|---|
| `LLM_MODEL` | `Qwen/Qwen3-14B-AWQ` (gptqmodel + Marlin kernel) |
| `embedding_model` | `BAAI/bge-m3` |
| `max_seq_length` | 256 |
| `K_LAW`, `K_COURT` | 200, 200 |
| `RRF k` | 60 |
| `graph_pool` | 100 |
| `graph_w` | 0.05 |
| **`out_n_law`** | **15** |
| **`out_n_court`** | **0** ← key insight |
| **`target_size`** | **17** (15 base + 2 LLM additional) |

## 🧪 Pseudo-val 검증 (train 100 → English translation)

LB best (sub_v7) ↔ pseudo-val best 정합 확인:

| config | val (10) | pseudo-val (100) | combined |
|---|---|---|---|
| **v7_oC0_t17_pG (LB best)** | 0.2088 | **0.0779** | **0.0898** ⭐ |
| v8_oC3_t20_gC_pG | 0.2214 | 0.0691 | 0.0830 |
| v8_oC2_t19_gC_pG | 0.2241 (val ↑) | 0.0721 | 0.0860 (over-fit) |

→ val에서 좋아 보인 graph C, larger output 등이 pseudo-val에서 일관 손해 = val 10개 over-fit이 정량 확인됨.

## 🔬 진단 인사이트

1. **임베딩이 진짜 병목** — 87% gold가 우리 top-200 retrieval pool 밖. LLM expansion으로 일부 메우지만 cross-lingual + abstraction gap 임베딩 단독으로 못 건넘.
2. **Court보다 BGE가 핵심** — gold court 60%가 BGE (leading authority cases). corpus 4%만이라 dense retrieval에서 묻힘.
3. **언어 장벽 작음** — multilingual encoder가 DE/FR/IT cross-lingual 매칭 25%/24%/거의 동등.
4. **abstraction gap이 진짜** — 같은 주제 case 100+ 중 정답 specific case 못 골라. dense retrieval 한계.
5. **Court output은 noise** — top-3 court까지만 신뢰. nC=0 (출력 안 함)이 best.

## ❌ 시도했으나 net negative였던 변형들 (16+ 종)

| 변형 | 결과 |
|---|---|
| BGE-reranker-v2-m3 (단독) | -0.056 |
| BGE-reranker-v2-m3 (hybrid 4종) | -0.002 ~ -0.041 |
| Round-2 iterative expansion | -0.0004 ~ +0.001 |
| Thinking mode expansion | -0.015 |
| Default Swiss codes union (kw 강제) | -0.009 |
| Per-code separate retrieval | -0.110 |
| Law neighbors (Art N → N±1) | -0.020 / +0.0004 |
| BM25 v2 with German kw | -0.039 |
| DiscoLM-German-7B expansion | -0.022 |
| Trans-only addition | -0.001 |
| Rejection loop (multi-turn LLM verdict) | -0.011 |
| Law-law co-citation graph | -0.010 ~ -0.060 |
| K_COURT 200→5000 (recall ↑ but graph noise) | -0.015 |
| Contextual augmentation (BGE 1067 sanity) | top-30 +5pp recall (작음) |
| Doctrinal augmentation A2 (rule+articles) | -0.044 (over-engineered) |
| Text gradient descent (5 iter prompt) | +0.004 (noise) |
| Graph C (court→BGE) on val 10 | val +0.013 BUT pval -0.009 (over-fit) |

## 📁 디렉토리 구조

```
legal-ir/
├── scripts/                       # 56 step-by-step scripts (00 ~ 56)
│   ├── 22_multiview_expand.py    # Multi-view expansion generator
│   ├── 25_test_submit_multiview.py
│   ├── 35_test_submit_planG.py   # 🏆 LB best submission generator
│   ├── 51_test_submit_combined.py
│   ├── 56_pseudoval_ablation.py   # Pseudo-val 검증
│   └── ...
├── expansions/                    # LLM-generated expansions (cached, JSON)
│   ├── val_expansions_qwen3_14b.json           # round-1
│   ├── val_multiview_qwen3_14b.json            # multi-view 6 views
│   ├── val_verify_reasoning_qwen3_14b.json     # Plan G additional
│   ├── test_expansions_qwen3_14b.json
│   ├── test_multiview_qwen3_14b.json
│   ├── test_verify_reasoning_qwen3_14b.json
│   ├── pseudo_val_*.json                        # 100 pseudo-val expansions
│   └── ...
├── submissions/                   # Submission CSVs + .meta.json
├── handoff/                       # Original handoff docs from prior agent session
├── SESSION_LOG.md                 # All Bash/Edit/Write commands logged automatically
├── log_hook.py                    # PostToolUse hook script
├── README.md                      # 이 파일
└── .gitignore                     # Excludes large artifacts (parquet, indexes, graph)
```

## 🛠 환경

- GPU: NVIDIA RTX 5090 32GB (sm_120, Blackwell)
- CUDA 12.8 (cu128 PyTorch wheel)
- Python 3.12 (Miniconda env `legal-ir`)
- transformers 5.8.0, gptqmodel 7.0.0, sentence-transformers 5.4.1, faiss-cpu 1.13.2

## 📦 큰 artifacts 재현

이 repo에 포함되지 않은 큰 파일들은 `scripts/`로 재빌드:

| Artifact | 크기 | 빌드 방법 |
|---|---|---|
| `parquet/{train,val,test,laws_de,court_considerations}.parquet` | 735MB | 원본 CSV → parquet (handoff/DATA_TRANSFER.md) |
| `indexes/bge_m3_v2_clean_fp16.npy` | 4.4GB | scripts/ 또는 PIPELINE_STATE.md (max_seq=256, length-sorted) |
| `indexes/{law,court}_v2.faiss` | 720MB + 8.1GB | IndexFlatIP, source-split |
| `graph/court_to_laws_v1.pkl` | 24MB | 정규식 추출, ~13초 |
| `graph/court_to_bges_v1.pkl` | 15MB | scripts/41_build_court_to_bges.py |

## 🚀 LB 제출 빌드 (best)

```bash
# Pre-requisite: parquet/, indexes/, graph/ 빌드되어 있어야
# (handoff/DATA_TRANSFER.md 또는 scripts/ 참고)

# 1. Round-1 expansion
python scripts/03_test_expansion.py

# 2. Multi-view expansion
python scripts/22_multiview_expand.py test

# 3. Verify reasoning (Plan G additional citations)
python scripts/34_test_verify_reasoning.py

# 4. Build submission (best config)
TARGET_SIZE=17 GRAPH_W=0.05 python scripts/35_test_submit_planG.py
# → submissions/sub_v7_planG_oL15_oC0_t17_*.csv
```

## 📊 진척 (대회 18-19일 → 2일)

- Day 1: 베이스라인들 (LLM 없이) 모두 LB 0.00679 이하
- Day 2 (이 프로젝트): Multi-view + Plan G로 **LB 0.15231 (+33% 상대 향상)**

## 📝 라이센스

Personal project for Kaggle competition. 코드는 reference용.
