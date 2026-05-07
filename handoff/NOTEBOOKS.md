# NOTEBOOKS — 기존 노트북 5개 설명

이전 환경에서 만들어진 5개 Colab 노트북. 새 서버로 옮기든, 참고용으로만 두든 자유.

---

## 01_setup_inventory.ipynb

**역할**: 데이터 다운로드 + parquet 변환 + 기본 통계.

**핵심 발견**:
- train: 1,139개 (논문에서 LEXam 4,886개라 했지만 우리에게 주어진 건 일부)
- laws_de: 175,933 (논문 15K 잘못)
- court_considerations: 2,476,315 (~2.43GB)
- val/test ID 형식: `val_001`, `test_001` (3자리)
- sample_submission ID 형식: `test_0001` (4자리) ← **호스트에서 수정한 듯, test_001 출력해도 채점됨 (앞선 제출에서 검증)**

**산출물**: `parquet/*.parquet`, `inventory.json`

**상태**: 완료, 산출물 캐시됨.

---

## 02_baseline.ipynb (실패 기록)

**시도**: BM25 + BGE-M3 (title prepend) hybrid + 통합 RRF + top-k

**문제**: 
- Title prepend가 임베딩 망침 (DBG 614개 조항이 모두 같은 헤더 시작 → 임베딩 mutual interference)
- BM25는 영어 query에 0 hits
- 합쳐서 sort 시 판례 점수가 항상 높아서 법령 묻힘

**val F1**: 0.000 (LB 점수 0.000)

**교훈**: prepend 절대 X, BM25 영어 query에 무용, 법령/판례 분리 retrieval 필수.

---

## 03_reembedding_v2.ipynb

**시도**: Clean text (prepend 제거) + max_seq=256 + length sort + 분리 인덱스

**핵심 변경**:
- Title/parent prepend 완전 제거
- max_seq 512 → 256 (시간 ½, 법령 평균 짧음)
- batch 128 → 256
- Length sort로 padding 낭비 제거
- 30자 미만 garbage text는 citation으로 대체

**산출물**:
- `indexes/bge_m3_v2_clean_fp16.npy` (~4GB)
- `indexes/law_v2.faiss`, `indexes/court_v2.faiss`
- `indexes/law_doc_ids_v2.pkl`, `indexes/court_doc_ids_v2.pkl`
- `indexes/bm25_v2/` (BM25 인덱스)

**val F1**: 0.0035 (약간 개선이지만 여전히 거의 0)

**진단된 한계**:
- val_001 gold 19개 법령 중 6개만 dense top-2000에 surface
- 나머지 13개는 임베딩으로 영원히 못 잡음
- 사실관계 → 추상 법령 abstraction gap

**상태**: 산출물 캐시됨. 임베딩 자체는 문제없으니 재빌드 X.

---

## 04_graph_1hop.ipynb

**시도**: 임베딩 위에 citation graph 1-hop expansion

**메커니즘**:
1. 200만 판례 본문에서 정규식으로 법령 인용 추출
2. `court_to_laws[court_id] = [law_ids]` 매핑
3. 검색 시 top-K 판례 → 본문 인용 법령 → 법령 후보에 boost

**Sanity 결과** (Art. 69 DBG, abstract keyword query):
- Top-30 판례 중 4개에 직접 인용 ✅

**Ablation 결과** (val 10개, 12 config):
- Dense-only best: 0.0073
- Dense + Graph best: 0.0094 (+0.002)

**진단**: 사실관계 query → 임베딩이 잡는 판례에 정답 법령 인용이 거의 없음 (cold start). Graph는 LLM expansion 후에야 의미 있을 것.

**산출물**:
- `graph/court_to_laws_v1.pkl`
- `graph/graph_stats_v1.json`

**상태**: 그래프 빌드 완료, 캐시됨.

---

## 05_agentic_expansion.ipynb (진행 중, 환경 문제로 미완료)

**시도**: Qwen3-14B AWQ로 query expansion → multi-query search

**설계**:
- 통합 prompt: `legal_areas`, `law_codes`, `keywords_de`, `hyde` 한 번에 JSON 출력
- Multi-query: 원본 + 키워드 + HyDE → RRF
- Graph 1-hop expansion 위에 얹음

**Colab 환경 문제로 막힘**:
- vLLM: ipykernel `fileno` 호환 버그
- transformers + AWQ: huggingface_hub `StrictDataclass` 충돌
- numpy `_center` import 에러
- ~3시간 dependency hell

**새 서버에서 재시도 예정** — RTX 5090 + 정상 Python 환경이면 문제 없을 것.

**산출물 (예정)**:
- `expansions/val_expansions_qwen3_14b.json`
- `expansions/test_expansions_qwen3_14b.json`
- `submissions/sub_v4_*.csv`

---

## 추천 실행 순서 (새 서버)

### Step 1: 환경 검증 (5분)
- `SETUP.md` 따라 패키지 설치
- 검증 코드로 import 정상 확인

### Step 2: 데이터/캐시 확보 (10-30분)
- `DATA_TRANSFER.md` 옵션 A (캐시 복사) 권장
- 검증 체크리스트 통과

### Step 3: Qwen3-14B AWQ 로드 검증 (5분)
- 단순 generation 테스트 (이전 환경에선 여기서 막혔음)
- 막히면 `PITFALLS.md` 참고하여 GGUF로 갈아타기

### Step 4: Agentic expansion 구현 (1-2시간)
- 05 노트북 로직 + `NEXT_STEPS.md`의 구체적 코드
- val 10개에 대한 expansion 생성
- ablation: orig only / kw only / hyde only / all

### Step 5: 결과 따라 분기
- val F1 > 0.05: reranker 추가 (06)
- val F1 ~0: prompt 튜닝, thinking mode, 큰 모델 (32B/MoE)
