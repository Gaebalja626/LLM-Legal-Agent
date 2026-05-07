# PITFALLS — 이전 환경에서 빠진 함정들 (반복 X)

이전 ~6시간 동안 빠진 함정들. 새 환경에선 같은 거 반복하지 마세요.

---

## 1. 환경 함정 (Colab 한정, 새 서버엔 영향 X 가능성 높음)

### 1.1 vLLM + ipykernel 호환 X
```
io.UnsupportedOperation: fileno
```
**원인**: vLLM이 stdout fd를 잡으려는데 ipykernel이 안 줌.
**새 서버에선**: 정상 Python 셸이면 OK. 그래도 transformers로 가는 게 안전.

### 1.2 transformers + AWQ + huggingface_hub 버전 충돌
```
StrictDataclassDefinitionError: Class 'Qwen3Config' must be a dataclass before applying @strict.
```
**원인**: `pip install --no-deps`로 깔아서 버전 mismatch.
**해결**: 절대 `--no-deps` 쓰지 말고 의존성 자동 해결 맡기기.

### 1.3 numpy `_center` import 에러
```
ImportError: cannot import name '_center' from 'numpy._core.umath'
```
**원인**: gptqmodel 설치 시 numpy 새 버전 끌고 와서 시스템 numpy의 컴파일된 빌드와 mismatch.
**해결**: 런타임 재시작 (importlib.reload 안 통함, C extension이라).

### 1.4 RTX 5090 sm_120 미지원 라이브러리
**잠재 위험**:
- `vllm`: sm_120 부분 지원 (transformers로 대체)
- `faiss-gpu`: sm_120 미지원 (faiss-cpu로)
- `autoawq`: 컴파일 안 되어있을 수 있음 (gptqmodel로)
- `flash-attn`: nightly만 지원

**전략**: 컴파일된 GPU 라이브러리 피하고 PyTorch + CPU 인덱스로. RTX 5090 32GB면 충분.

---

## 2. 데이터 함정

### 2.1 Title/parent prepend 절대 금지
**증상**: DBG 614개 조항이 모두 같은 헤더 (`"Bundesgesetz vom 14. Dezember 1990 über die direkte Bundessteuer..."`)로 시작 → 임베딩 공간에서 mutual interference → query랑 멀어짐.

**대신**: title은 별도 컬럼으로 보관하고 검색 후처리에 활용.

### 2.2 BM25 + 영어 query = 무용
val_001 영어 query → BM25 top-200에 0 hits.
**해결**: BM25는 LLM이 만든 독일어 키워드 query에만 적용.

### 2.3 합친 source ranking
법령 cosine: 0.55-0.61, 판례 cosine: 0.66-0.68. 합쳐서 sort하면 법령 묻힘.
**해결**: source별 분리 retrieval + quota 강제.

### 2.4 점수 스케일 비교 (RRF에서)
```python
[0.0164] doc_a
[0.0164] doc_b
[0.0161] doc_c
```
이렇게 점수가 다 1/(60+rank+1) 같으면 = **두 ranking이 disjoint**라는 뜻. 즉 BM25랑 Dense가 완전 다른 결과 내고 있음 → BM25 노이즈.

### 2.5 split_cites 후 set 변환의 순서
```python
list(set(cites))[:5]  # 순서 비결정적, 매번 다를 수 있음
```
디버깅 시 헷갈리지 말기. 비교는 항상 set 연산으로.

---

## 3. 임베딩 함정

### 3.1 max_seq=512 padding 낭비
법령 평균 ~200-300자, max_seq=512면 패딩에 시간 낭비.
**해결**: max_seq=256 + length sort.

### 3.2 임베딩 unsort 로직 실수
```python
sort_idx = np.argsort(lengths)
unsort_idx = np.argsort(sort_idx)
embeddings = emb_sorted[unsort_idx]  # 원래 순서로
```
이거 잘못하면 인덱스가 어긋나서 검색 결과 무작위. 사용자 v2 빌드는 검증됨 (cosine 0.999996).

### 3.3 garbage text (`"und"`, `"."` 같은 짧은 row)
court_considerations에 30자 미만 ~5%. 그대로 임베딩하면 noise.
**해결**: 짧은 text는 citation으로 대체 (인덱스 row 보존, 검색에선 잘 안 잡힘).

---

## 4. 평가 함정

### 4.1 val 10개에 오버핏 위험
val 너무 작아서 1개 query 차이로 F1 0.1 변동. 일반화 신뢰 X.
**대응**: train 일부를 영어로 번역해 pseudo-val 200개 만들기 (시간 되면).

### 4.2 train F1 신뢰 X
train 독일어, val/test 영어. train F1 높아도 transfer 안 될 수 있음.

### 4.3 출력 개수 캘리브레이션 = F1의 절반
val 평균 gold 25, train 평균 4. 고정 top-k로 망함. 가변 출력 필수.

### 4.4 LB public 20개 오버핏
대회 LB 점수는 public 20개 기준. 이걸로 튜닝 시 private 20개에서 망할 수 있음.
**대응**: val + 가능하면 pseudo-val로 튜닝, LB는 sanity check만.

---

## 5. 프로세스 함정 (사용자 피드백 반영)

### 5.1 "베이스라인부터" 명분으로 사용자 직관 무시
사용자가 처음부터 "graph + agentic이 핵심"이라고 정확히 가리켰음. 이전 에이전트(나)가 5번 미루고 결국 같은 결론 도달. **사용자 직관 신뢰**.

### 5.2 매번 "더 작은 ablation 먼저"
이미 10개 ablation 한 사람한테 또 더 검증하라고 시키지 말 것. 결정 패턴 보이면 바로 진도.

### 5.3 vLLM처럼 빌드 무거운 라이브러리 권장
사용자가 결국 한 번 더 setup 거쳐야 됨. transformers처럼 단순한 거가 디버깅 쉬움.

### 5.4 토픽 분산
한 번에 한 가지. 임베딩 빌드 끝나기 전에 graph 빌드 같이 해라 같은 거 X.

---

## 6. 만약 또 막히면

### 6.1 dependency hell이 또 일어나면
1. 패키지 설치 history 메모
2. **즉시 GGUF로 갈아타기** (llama-cpp-python, 의존성 거의 없음):
```python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen3-14B-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
)
```

### 6.2 모델이 너무 커서 안 들어가면
RTX 5090 32GB 기준:
- BGE-M3 (~2GB) + reranker (~2GB) + Qwen3-14B AWQ (~9GB) = 13GB ✅
- + Qwen3-32B AWQ (~19GB) → 동시 로드 X, swap 필요
- Qwen3-30B-A3B MoE 4bit (~18GB) + 임베딩들 → 빠듯

**전략**: LLM과 임베딩 모델 swap. Expansion 단계에서 LLM 로드, 검색 단계에서 임베딩 로드.

### 6.3 LLM 추론이 느리면
- batch 키우기
- max_new_tokens 줄이기 (1500 → 800)
- thinking mode 끄기 (`enable_thinking=False`)
- 양자화 더 (AWQ-3bit 또는 GGUF Q3_K_M)
