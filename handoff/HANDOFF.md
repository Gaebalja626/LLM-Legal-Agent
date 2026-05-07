# Handoff: Swiss Legal IR Competition

> **For the next AI agent**: 이 문서를 처음부터 끝까지 읽으세요. 사용자가 ~6시간 분량의 진단을 거쳤고, 그 결과로 솔루션 방향이 거의 확정된 상태입니다. 같은 함정을 반복하지 마세요.

---

## 1. 대회 한 줄 요약

- **Kaggle**: "LLM Agentic Legal Information Retrieval" (Omnilex 후원)
- **Task**: 영어 법률 질문 40개 → 스위스 법률 코퍼스에서 인용(citation) 집합 retrieval
- **평가**: Macro F1 (per-query F1 평균)
- **마감**: 2026-05-06 기준 약 18-19일 남음
- **상금**: 총 $10,000 (1st $5,000)
- **제출**: Kaggle 노트북, 오프라인 12시간 이내 실행, 외부 API 금지

자세한 내용은 `OVERVIEW.md` 참고.

---

## 2. 새 환경 (지금 작업할 곳)

```
GPU:    NVIDIA RTX 5090 (32GB VRAM)
CUDA:   13.2
Driver: 595.79
```

**이전 환경 (Colab + RTX PRO 6000 96GB)에서 ipykernel + vLLM/transformers + numpy dependency hell 때문에 모델 로드 실패 ~3시간 소비**. 새 서버는 정상 Python 환경이라 그 이슈 없을 것으로 예상.

**32GB VRAM 제약**:
- Qwen3-14B AWQ-4bit: 약 9GB ← 추천 사이즈
- BGE-M3: 약 2GB
- BGE-reranker-v2-m3: 약 2GB
- 동시 로드 가능 (총 ~13GB)
- Qwen3-32B AWQ는 ~19GB라 임베딩 모델과 동시 로드 시 빠듯, swap 권장
- Qwen3-30B-A3B MoE 4-bit는 ~18GB, active 3B라 빠름 — **이게 sweet spot일 가능성**

---

## 3. 드라이브에 캐시된 산출물 (재계산 X)

이전 환경에서 빌드해둔 게 사용자 Google Drive `LLM Legal Agent/` 폴더에 있습니다. 새 서버에 옮겨야 합니다.

```
LLM Legal Agent/
├── parquet/
│   ├── train.parquet              (1,139 queries, 독일어, 670KB)
│   ├── val.parquet                (10 queries, 영어, 20KB)
│   ├── test.parquet               (40 queries, 영어, 30KB)
│   ├── laws_de.parquet            (175,933 법령, 13MB)
│   ├── court_considerations.parquet (2,476,315 판례, 720MB)
│   ├── sample_submission.parquet
│   └── inventory.json
├── indexes/
│   ├── bge_m3_v2_clean_fp16.npy   (~4GB, 2.16M × 1024 fp16)
│   ├── law_v2.faiss               (법령 FAISS IndexFlatIP)
│   ├── court_v2.faiss             (판례 FAISS IndexFlatIP)
│   ├── law_doc_ids_v2.pkl
│   ├── court_doc_ids_v2.pkl
│   └── bm25_v2/                   (BM25 인덱스)
├── graph/
│   ├── court_to_laws_v1.pkl       (판례→인용법령 매핑)
│   └── graph_stats_v1.json
└── submissions/
    └── 여러 시도 결과들 (val F1 모두 ~0)
```

**복사 방법**: 사용자가 드라이브에서 직접 다운로드 → 새 서버로 scp/rsync. 또는 사용자가 그냥 모든 캐시를 zip해서 서버에 업로드.

원본 데이터 (zip ~789MB)는 사용자 드라이브에 `llm-agentic-legal-information-retrieval.zip` 으로 있음. 캐시 옮기기 귀찮으면 raw에서 다시 빌드 가능 (시간: setup 2분 + 임베딩 ~30분).

---

## 4. 지금까지 진단된 핵심 사실 (재검증 X)

### 4.1 데이터 특성

- **train (1,139)**: 독일어 query, citation 평균 4개 (median 2)
- **val (10)**: 영어 query, citation 평균 25개 (median 22, max 47)
- **test (40)**: 영어 query, citation 평균 ?? (라벨 없음)
- **train 분포가 val/test와 완전히 다름** — train F1 신뢰 X
- val gold의 100%가 corpus에 존재 (oracle ceiling = 1.0)
- train gold의 71%만 corpus에 (28.9%는 article 단위로 적혀있는데 corpus엔 항(Abs.) 단위로만 있음)

### 4.2 Citation 형식

```
법령:     Art. N XYZ            (예: Art. 1 ZGB)
         Art. N Abs. M XYZ     (예: Art. 11 Abs. 2 OR)
판례:     BGE V S P E. N        (예: BGE 145 II 32 E. 3.1)
         5A_800/2019 E 2.       (docket 형식)
         2P.260/2003 13.01.2004 E. 1  (옛 docket, 점 포함)
```

**정확 매칭 필수**: corpus citation 문자열과 한 글자도 다르면 false positive.

### 4.3 임베딩의 본질적 한계 (검증됨)

BGE-M3 v2 (clean text, max_seq=256, length-sorted) 기준:

```
val_001 query (사실관계 narrative, gold 42개):
  법령 hits in dense top-2000:  6 / 19개 추정
  나머지 13개 법령은 top-2000 밖 — 임베딩으로 영원히 못 잡음
  
  법령 cosine 점수: 0.55-0.61
  판례 cosine 점수: 0.66-0.68 (항상 더 높음)
  → 합쳐서 sort하면 법령 묻힘
```

**원인**: 사실관계(1500자 영어 narrative) → 추상 법령(독일어 조문)의 abstraction gap. 임베딩만으로는 못 건넘.

**검증**: 같은 query를 hand-crafted abstract 독일어 query (`"Schiedsvereinbarung internationale Zuständigkeit IPRG"` 같은)로 바꾸면 hits 0/4 → 2/4로 회복. **즉 LLM이 사실관계 → 법률 개념 변환 단계가 핵심**.

### 4.4 Graph (1-hop citation) 효과 (검증됨)

판례 본문에서 정규식으로 법령 인용 추출 → vocabulary 매칭. 200만 판례 처리 ~10-20분.

**Sanity 결과** (`Art. 69 DBG`, abstract keyword query):
- Top-30 판례 중 4개에 직접 인용 (rank 2, 3, 10, 20)
- 추출 정규식: `Art\.\s*(\d+\w*)(?:\s*Abs\.\s*(\d+\w*))?\s*([A-Z][\w\-\.]+)`

**문제**: 임베딩 자체가 약하면 graph도 cold start. val_001 같은 사실관계 query는 임베딩이 잡는 판례에 정답 법령 인용 거의 없음.

**Ablation 결과** (config 12개 스윕):
```
Dense-only best F1:        0.0073
Dense + Graph best F1:     0.0094  (+0.002, 노이즈 수준)
```

→ Graph는 **LLM expansion 후에야** 의미 있을 가능성 (LLM이 의미 있는 후보 surface하면 그래프가 증폭).

### 4.5 BM25는 영어 query에 무용 (검증됨)

영어 query → 독일어 BM25 매칭 거의 0 hits. RRF에 BM25 넣어도 noise만 추가. **베이스라인에서 BM25 제거 권장**.

만약 BM25 쓰려면 LLM이 만든 독일어 키워드 query에 적용.

---

## 5. 솔루션 방향 (확정)

```
Query (English narrative, ~1500자)
  ↓
[STAGE 1] LLM agentic expansion (Qwen3-14B AWQ)
  ├── Step 1: 법적 쟁점 분류 (legal_areas)
  ├── Step 2: 관련 법전 식별 (law_codes: StPO, OR, ZGB 등)
  ├── Step 3: 독일어 법률 키워드 (keywords_de)
  └── Step 4: HyDE — 가상 독일어 답변 (hyde)
  → JSON 출력
  ↓
[STAGE 2] Multi-query dense retrieval (BGE-M3, split 법령/판례)
  ├── 원본 query
  ├── keywords + law_codes joined
  └── HyDE
  → 각각 검색 → RRF
  ↓
[STAGE 3] Graph 1-hop expansion
  → 잡힌 판례 본문에서 법령 인용 추출 → 법령 후보 boost
  ↓
[STAGE 4] (선택) Cross-encoder reranker (BGE-reranker-v2-m3)
  → top-N → 정밀 ranking
  ↓
[STAGE 5] Output
  → 법령 N_L + 판례 N_C quota 강제
  → val에서 N_L, N_C 튜닝
  → submission.csv
```

이게 *대회 이름이 "Agentic Legal IR"인 이유*입니다. LLM 없이는 본질적으로 못 풉니다.

---

## 6. 즉시 다음 단계

1. **환경 셋업** (`SETUP.md` 참고)
2. **이전 데이터/인덱스 복사** 또는 raw에서 재빌드
3. **Qwen3-14B AWQ 로드 검증** — 이게 새 환경에서도 막히면 빨리 GGUF로 갈아타기 (`PITFALLS.md`)
4. **05_agentic_expansion 노트북 실행** (`NOTEBOOKS.md`)
5. **결과 따라 분기**:
   - val F1 > 0.05: reranker 추가 (06)
   - val F1 ~0: thinking mode 켜고 재시도 / 모델 32B로 업스케일 / prompt 개선

---

## 7. 사용자에 대한 노트

- 한국어 사용. 직설적이고 욕도 가끔 합니다 (스트레스 표현, 상호 비난 아님). 정상 응대.
- 도메인 직관 강합니다. **graph + agentic 방향**을 처음부터 정확히 가리켰는데 이전 에이전트(저)가 데이터 검증 명분으로 5번 미뤘습니다. **사용자 직관 신뢰하세요**.
- 효율적 진행 선호. 코드 길게 설명하기보다 짧게 코드 주고 실행, 결과 보고 다음 코드.
- "재임베딩하면 다시 30분 걸리잖아"처럼 **시간 비용에 민감**. 캐시 활용, 불필요한 재계산 X.
- 결정 필요 시 `ask_user_input_v0` 같은 멀티초이스 형태 좋아함.

---

## 8. 파일 가이드

이 핸드오프 폴더의 다른 문서들:
- `SETUP.md` — 새 서버 환경 셋업 절차
- `DATA_TRANSFER.md` — 이전 캐시 옮기기 또는 raw에서 재빌드
- `PIPELINE_STATE.md` — 현재까지 구축된 파이프라인 상세 (재현용 코드 포함)
- `NOTEBOOKS.md` — 5개 기존 노트북 설명 + 실행 순서
- `PITFALLS.md` — 이전 환경에서 겪은 함정 (다시 빠지지 말 것)
- `NEXT_STEPS.md` — 우선순위 액션 아이템
