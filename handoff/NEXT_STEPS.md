# NEXT_STEPS — 즉시 다음 액션 + 우선순위

## 0순위: 환경 검증 (5분)

`SETUP.md` 따라 셋업 후, **간단한 LLM 로드 테스트** 먼저:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_MODEL = 'Qwen/Qwen3-14B-AWQ'

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    dtype=torch.float16,
    device_map='cuda',
)
model.eval()

# Test generation
msg = [{"role": "user", "content": "Hello, please say hi back in one word."}]
prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Output: {text}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

**막히면**: `PITFALLS.md` 6.1번 참고하여 GGUF로 갈아타기. 여기서 ~30분 이상 걸리면 빠르게 결단.

---

## 1순위: Agentic Expansion 구현 (1-2시간)

### 1.1 Expansion 프롬프트

이전 환경에서 만들어둔 거 그대로 사용:

```python
EXPANSION_SYSTEM = """You are an expert in Swiss law. You will analyze a legal question (in English, often a long fact pattern) and produce structured search aids in German.

Your task: identify legal issues, extract German legal keywords, and write a hypothetical German legal analysis.

Output STRICTLY valid JSON with these keys:
{
  "legal_areas": [list of relevant Swiss legal areas, e.g. "Strafverfahrensrecht", "Obligationenrecht"],
  "law_codes": [list of Swiss law abbreviations likely relevant, e.g. "StPO", "OR", "ZGB", "StGB", "BV", "BGG", "IPRG", "DBG"],
  "keywords_de": [German legal keywords/phrases, 5-15 items, specific legal terms not generic words],
  "hyde": "A hypothetical 200-400 word German legal analysis answering the question, using precise legal terminology. Cite article numbers when natural. Write as if for a legal opinion."
}

Output ONLY the JSON, no other text."""

def build_prompt(query):
    messages = [
        {'role': 'system', 'content': EXPANSION_SYSTEM},
        {'role': 'user', 'content': f'Legal question:\n\n{query}\n\nProduce the JSON now.'},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
```

### 1.2 Batch generation

```python
import json, re

@torch.inference_mode()
def llm_generate_batch(prompts, max_new_tokens=1500, temperature=0.2, batch_size=4):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, 
                          truncation=True, max_length=4096).to('cuda')
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for j, out in enumerate(outputs):
            input_len = inputs['input_ids'][j].shape[0]
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            results.append(text)
        print(f'  {i+len(batch)}/{len(prompts)}')
    return results

def parse_json_loose(text):
    """JSON 추출 시도 (LLM이 ``` 안에 넣거나 prefix 텍스트 붙일 수 있음)"""
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        pass
    return None
```

### 1.3 Val expansion 생성 + 캐시

```python
val = pd.read_parquet(PARQUET_DIR / 'val.parquet')
val['cites'] = val['gold_citations'].apply(split_cites)

VAL_EXP_PATH = EXP_DIR / 'val_expansions_qwen3_14b.json'

if VAL_EXP_PATH.exists():
    with open(VAL_EXP_PATH) as f:
        val_expansions = json.load(f)
else:
    prompts = [build_prompt(q) for q in val['query'].tolist()]
    print(f'Val expansion 생성 ({len(prompts)}개)')
    t0 = time.time()
    texts = llm_generate_batch(prompts)
    print(f'Done: {time.time()-t0:.1f}s')
    
    val_expansions = {}
    for vq, text in zip(val.itertuples(), texts):
        parsed = parse_json_loose(text)
        val_expansions[vq.query_id] = {'parsed': parsed, 'raw': text}
    
    with open(VAL_EXP_PATH, 'w') as f:
        json.dump(val_expansions, f, indent=2, ensure_ascii=False)
```

### 1.4 Expansion 품질 점검

```python
v1 = val_expansions['val_001']
print(json.dumps(v1['parsed'], indent=2, ensure_ascii=False))
print('\nGold 법전들:', set(c.split()[-1] for c in val.iloc[0]['cites']))
# 기대: StPO, StGB, BGG, BV 등이 law_codes에 있어야
```

### 1.5 Multi-query search

```python
# (LLM 메모리 해제 → 임베딩 모델 로드, 32GB라 swap 안전)
del model
gc.collect()
torch.cuda.empty_cache()

# (PIPELINE_STATE.md의 search_dense_split, court_to_laws 로드)

def search_with_expansion(query, expansion, 
                          k_law=150, k_court=150,
                          graph_pool=100, graph_w=0.05,
                          out_n_law=25, out_n_court=15):
    # Build query variants
    queries = [query]  # 원본
    if isinstance(expansion, dict):
        kws = expansion.get('keywords_de', [])
        codes = expansion.get('law_codes', [])
        if kws or codes:
            queries.append(' '.join(codes + kws))  # 키워드 query
        hyde = expansion.get('hyde', '')
        if hyde and len(hyde) > 50:
            queries.append(hyde)  # HyDE
    
    # Search each → RRF
    law_rankings, court_rankings = [], []
    for q in queries:
        laws, courts = search_dense_split(q, k_law=k_law, k_court=k_court)
        law_rankings.append(laws)
        court_rankings.append(courts)
    
    law_scores = rrf_merge_dicts(law_rankings)
    court_scores = rrf_merge_dicts(court_rankings)
    
    # Graph boost
    court_top = sorted(court_scores.items(), key=lambda x: -x[1])
    graph_freq = Counter()
    for d, _ in court_top[:graph_pool]:
        for law in court_to_laws.get(d, []):
            graph_freq[law] += 1
    
    for law, freq in graph_freq.items():
        law_scores[law] = law_scores.get(law, 0.0) + freq * graph_w
    
    # Output
    law_top = sorted(law_scores.items(), key=lambda x: -x[1])[:out_n_law]
    return [d for d, _ in law_top] + [d for d, _ in court_top[:out_n_court]]
```

### 1.6 Val ablation

```python
golds = val['cites'].tolist()

# 모드 ablation
for mode_name, modes in [
    ('orig only',     ['orig']),
    ('kw only',       ['kw']),
    ('hyde only',     ['hyde']),
    ('orig + kw',     ['orig', 'kw']),
    ('orig + hyde',   ['orig', 'hyde']),
    ('kw + hyde',     ['kw', 'hyde']),
    ('all',           ['orig', 'kw', 'hyde']),
]:
    preds = []
    for vq in val.itertuples():
        exp = val_expansions[vq.query_id]['parsed']
        # 모드에 따라 query 빌드
        ...
        preds.append(search_with_expansion(...))
    f1 = macro_f1(preds, golds)
    print(f'  {mode_name}: F1={f1:.4f}')
```

### 1.7 Output size 튜닝

가장 좋은 mode 고정 후 (out_n_law, out_n_court) 스윕.

### 1.8 Test 예측

LLM 다시 로드 → test 40개 expansion → 검색 → submission.

---

## 2순위: 결과 분기

### val F1 > 0.10
**다음**: Reranker 추가 (06)
- BGE-reranker-v2-m3 (~2GB)
- Top-100 candidates → cross-encoder rerank → top-25
- 기대 효과: +0.05~0.10

### val F1 0.05~0.10
**다음 옵션**:
1. Thinking mode 켜고 expansion 재생성 (`enable_thinking=True`, 시간 2-3배)
2. 모델 Qwen3-32B AWQ로 업스케일 (~19GB, 32GB GPU에 swap)
3. Prompt iteration (legal_areas/law_codes/keywords_de 비중 조정)

### val F1 < 0.05
**의심 사항**:
- LLM이 JSON parsing fail (raw 출력 점검)
- Expansion이 의미 없는 generic 키워드만 (prompt 강화)
- 검색 단계 버그 (search_with_expansion 단위 테스트)

---

## 3순위: 추가 개선 (시간 되면)

### 3.1 Pseudo-val 만들기
- Train 1,139개 중 200개를 영어로 번역 (Qwen3-14B로)
- Val 10개 + pseudo-val 200개로 신뢰성 있는 튜닝
- 시간: 번역 ~10분 + 평가 ~5분

### 3.2 Reranker fine-tuning
- BGE-reranker-v2-m3을 train 데이터로 fine-tune
- (query, gold citation) → positive pair
- Hard negative mining
- 시간: 1-2시간

### 3.3 더 풍부한 graph
- 법령 → 인접 조항 (Art. N → Art. N+1)
- 같은 BGE 내 considerations 연결
- 양방향 (법령 → 인용한 판례)

### 3.4 Iterative agentic loop
- Round 1: expansion + 검색 → top-30
- Round 2: LLM이 결과 보고 부족한 측면 식별 → 추가 expansion
- Round 3: 최종 verifier
- 시간: 노트북 1개로 1-2시간

---

## 4순위: Kaggle 제출 준비

대회 마감 18-19일 남음. 마지막 3일은 Kaggle 노트북 변환 + 오프라인 검증에 써야 함:

1. **모델/인덱스를 Kaggle Dataset으로 업로드**
   - BGE-M3, Qwen3-14B AWQ, FAISS 인덱스, graph
2. **Kaggle 노트북 작성** (인터넷 없이 실행)
   - 모든 모델/데이터 path를 `/kaggle/input/...`으로
3. **12시간 안에 완료 검증**
   - 40개 query × LLM expansion + 검색 = ~10-30분 예상
4. **Submission 파일 자동 생성 검증**

---

## 사용자에게 결과 보고할 때

각 단계 후:
- val F1 (전체 + per-query)
- 최적 config
- 어디서 막혔는지 (있으면)
- 다음 제안

수치만 보고하지 말고 짧은 해석 같이.
