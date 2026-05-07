# PIPELINE_STATE — 현재까지 구축된 파이프라인 (재현용)

이 문서는 노트북 01-04에서 만들어진 파이프라인의 **재현 가능한 통합 코드**입니다.
캐시가 다 있으면 그냥 읽기만 하고, raw에서 빌드 시 이 코드를 그대로 쓰세요.

---

## 0. 공통 import + 경로

```python
from pathlib import Path
import json, time, pickle, gc, re
from collections import Counter, defaultdict
import pandas as pd
import polars as pl
import numpy as np
import torch

P = Path('~/legal-ir').expanduser()
DATA_DIR = P / 'data'
PARQUET_DIR = P / 'parquet'
INDEX_DIR = P / 'indexes'
GRAPH_DIR = P / 'graph'
SUB_DIR = P / 'submissions'
EXP_DIR = P / 'expansions'
for d in [INDEX_DIR, GRAPH_DIR, SUB_DIR, EXP_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

---

## 1. 평가 함수 (필수)

```python
def split_cites(s):
    if pd.isna(s) or s == '':
        return []
    return [x.strip() for x in s.split(';') if x.strip()]

def f1_per_query(pred, gold):
    pred_set, gold_set = set(pred), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(gold_set)
    return 2 * p * r / (p + r)

def macro_f1(preds, golds):
    return float(np.mean([f1_per_query(p, g) for p, g in zip(preds, golds)]))
```

---

## 2. Corpus 재구성 (clean text, prepend 없음)

⚠️ **이전 v1 시도에서 title prepend가 임베딩 망쳤음.** v2에선 절대 prepend 하지 마세요.

```python
laws_raw = pd.read_parquet(PARQUET_DIR / 'laws_de.parquet')
court_raw = pl.read_parquet(PARQUET_DIR / 'court_considerations.parquet').to_pandas()

# 법령: 본문만 (title은 별도 컬럼으로 보관 — 검색 후처리에 활용 가능)
laws_clean = pd.DataFrame({
    'citation': laws_raw['citation'],
    'text': laws_raw['text'].astype(str),
    'title': laws_raw['title'].astype(str),
    'source': 'law',
})

# 판례: 본문만
court_clean = pd.DataFrame({
    'citation': court_raw['citation'],
    'text': court_raw['text'].astype(str),
    'title': '',
    'source': 'court',
})

corpus = pd.concat([laws_clean, court_clean], ignore_index=True)

# Dedup (같은 citation 여러 row면 text concat, 3000자 cap)
corpus = (corpus.groupby('citation', as_index=False)
          .agg({'text': lambda xs: ' '.join(xs.astype(str))[:3000],
                'title': 'first',
                'source': 'first'}))

# Garbage 처리: 30자 미만 text는 citation으로 대체 (인덱스 row 보존)
corpus['text_len'] = corpus['text'].str.len()
mask_short = corpus['text_len'] < 30
corpus.loc[mask_short, 'text'] = corpus.loc[mask_short, 'citation']

del laws_raw, court_raw, laws_clean, court_clean
gc.collect()

print(f'corpus: {len(corpus):,}')
print(corpus['source'].value_counts())
# 기대: law 175,933  /  court 1,985,178  (총 ~2.16M)
```

---

## 3. BGE-M3 임베딩 빌드 (캐시 없을 때만)

```python
from sentence_transformers import SentenceTransformer

EMB_PATH = INDEX_DIR / 'bge_m3_v2_clean_fp16.npy'

if EMB_PATH.exists():
    print('임베딩 캐시 로드')
    embeddings = np.load(EMB_PATH).astype('float32')
    print(f'shape: {embeddings.shape}')
else:
    print(f'임베딩 빌드 ({len(corpus):,} docs, ~30-60분)')
    t0 = time.time()
    
    model = SentenceTransformer('BAAI/bge-m3', device='cuda')
    model.max_seq_length = 256  # 짧게 두면 빠름. 효과 없으면 512로 늘려보기
    
    texts = corpus['text'].tolist()
    
    # Length sort로 padding 낭비 줄임 (key trick)
    lengths = np.array([len(t) for t in texts])
    sort_idx = np.argsort(lengths)
    unsort_idx = np.argsort(sort_idx)
    texts_sorted = [texts[i] for i in sort_idx]
    
    chunks = []
    CHUNK = 100_000
    BATCH = 256  # RTX 5090에선 384도 가능, OOM 나면 줄이기
    
    for i in range(0, len(texts_sorted), CHUNK):
        ck = texts_sorted[i:i+CHUNK]
        emb = model.encode(
            ck,
            batch_size=BATCH,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        chunks.append(emb.astype('float16'))
        elapsed = time.time() - t0
        done = i + len(ck)
        eta = elapsed / done * (len(texts_sorted) - done)
        print(f'  {done:,}/{len(texts_sorted):,} | {elapsed:.0f}s | ETA: {eta:.0f}s')
    
    emb_sorted = np.vstack(chunks)
    embeddings = emb_sorted[unsort_idx]  # 원래 corpus 순서로 복원
    np.save(EMB_PATH, embeddings.astype('float16'))
    embeddings = embeddings.astype('float32')
    
    del chunks, emb_sorted, model
    torch.cuda.empty_cache()
    gc.collect()
    print(f'전체: {time.time()-t0:.1f}s')
```

---

## 4. FAISS 인덱스 (분리: 법령/판례)

⚠️ **반드시 분리.** 이전 진단에서 합친 검색은 판례 점수가 항상 높아서 법령 묻힘.

```python
import faiss

laws_mask = (corpus['source'] == 'law').values
court_mask = (corpus['source'] == 'court').values

LAW_DOC_IDS = corpus.loc[laws_mask, 'citation'].tolist()
COURT_DOC_IDS = corpus.loc[court_mask, 'citation'].tolist()

law_index = faiss.IndexFlatIP(embeddings.shape[1])
law_index.add(embeddings[laws_mask])

court_index = faiss.IndexFlatIP(embeddings.shape[1])
court_index.add(embeddings[court_mask])

# 저장
faiss.write_index(law_index, str(INDEX_DIR / 'law_v2.faiss'))
faiss.write_index(court_index, str(INDEX_DIR / 'court_v2.faiss'))
with open(INDEX_DIR / 'law_doc_ids_v2.pkl', 'wb') as f:
    pickle.dump(LAW_DOC_IDS, f)
with open(INDEX_DIR / 'court_doc_ids_v2.pkl', 'wb') as f:
    pickle.dump(COURT_DOC_IDS, f)

print(f'law_index: {law_index.ntotal:,}')
print(f'court_index: {court_index.ntotal:,}')
```

---

## 5. 검색 함수 (split dense)

```python
MODEL = SentenceTransformer('BAAI/bge-m3', device='cuda')
MODEL.max_seq_length = 256

def search_dense_split(query, k_law=200, k_court=200):
    """법령/판례 따로 dense 검색. k=0 처리 포함."""
    q = MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype('float32')
    laws, courts = [], []
    if k_law > 0:
        s_l, i_l = law_index.search(q, k_law)
        laws = [(LAW_DOC_IDS[idx], float(s)) for idx, s in zip(i_l[0], s_l[0])]
    if k_court > 0:
        s_c, i_c = court_index.search(q, k_court)
        courts = [(COURT_DOC_IDS[idx], float(s)) for idx, s in zip(i_c[0], s_c[0])]
    return laws, courts
```

---

## 6. Citation Graph (1-hop, 정규식)

```python
GRAPH_PATH = GRAPH_DIR / 'court_to_laws_v1.pkl'
STATS_PATH = GRAPH_DIR / 'graph_stats_v1.json'

# 법령 vocabulary로 정확 매칭
law_vocab = set(LAW_DOC_IDS)

# 정규식: Art. X (Abs. Y)? XYZ
CITE_PATTERN = re.compile(
    r'Art\.\s*(\d+[a-zA-Z]*(?:bis|ter|quater)?)'
    r'(?:\s*Abs\.\s*(\d+[a-zA-Z]*))?'
    r'\s*([A-Z][A-Za-z\-]+(?:\.[A-Z][A-Za-z\-]+)*)'
)

def extract_law_citations(text, vocab):
    found = set()
    for m in CITE_PATTERN.finditer(text):
        art = m.group(1)
        abs_ = m.group(2)
        abbrev = m.group(3)
        cit = f'Art. {art} Abs. {abs_} {abbrev}' if abs_ else f'Art. {art} {abbrev}'
        if cit in vocab:
            found.add(cit)
    return found

if GRAPH_PATH.exists():
    print('graph 캐시 로드')
    with open(GRAPH_PATH, 'rb') as f:
        court_to_laws = pickle.load(f)
else:
    print('graph 빌드 (~10-15분)')
    t0 = time.time()
    court_rows = corpus[corpus['source'] == 'court']
    court_to_laws = {}
    citations_list = court_rows['citation'].tolist()
    texts_list = court_rows['text'].tolist()
    
    for i in range(0, len(citations_list), 100_000):
        for cit, txt in zip(citations_list[i:i+100_000], texts_list[i:i+100_000]):
            laws = extract_law_citations(txt, law_vocab)
            if laws:
                court_to_laws[cit] = list(laws)
        elapsed = time.time() - t0
        print(f'  {min(i+100_000, len(citations_list)):,}/{len(citations_list):,} | {elapsed:.0f}s')
    
    with open(GRAPH_PATH, 'wb') as f:
        pickle.dump(court_to_laws, f)
    
    stats = {
        'total_court': len(court_rows),
        'court_with_law_citations': len(court_to_laws),
        'build_time_sec': time.time() - t0,
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(stats)
```

---

## 7. 통합 검색 함수 (LLM 없이도 작동)

```python
def rrf_merge_dicts(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking):
            scores[doc] += 1.0 / (k + rank + 1)
    return scores

def search_graph_aug(query, 
                     dense_k_law=80, dense_k_court=80,
                     graph_court_pool=50,
                     graph_freq_weight=0.05,
                     out_n_law=20, out_n_court=15):
    """
    LLM 없이 dense + graph 조합. baseline.
    val에서 0.0094 정도 나옴 — 거의 노이즈 수준.
    """
    dense_laws, dense_courts = search_dense_split(
        query, k_law=dense_k_law, k_court=max(dense_k_court, graph_court_pool)
    )
    
    law_scores = {d: s for d, s in dense_laws}
    
    # Graph: top pool 판례에서 인용된 법령 빈도
    graph_freq = Counter()
    for d, _ in dense_courts[:graph_court_pool]:
        for law in court_to_laws.get(d, []):
            graph_freq[law] += 1
    
    for law, freq in graph_freq.items():
        boost = freq * graph_freq_weight
        law_scores[law] = law_scores.get(law, 0.0) + boost
    
    sorted_laws = sorted(law_scores.items(), key=lambda x: -x[1])
    final_laws = [d for d, _ in sorted_laws[:out_n_law]]
    final_courts = [d for d, _ in dense_courts[:out_n_court]]
    
    return final_laws + final_courts
```

이 함수만 쓰면 val F1 ≈ 0.01. **이 위에 LLM expansion 얹어야 점수 나옴.** → `NEXT_STEPS.md`

---

## 8. 시도된 베이스라인들 (모두 실패)

`submissions/` 폴더 참고. 시간순:

| 시도 | 파이프라인 | val F1 | 비고 |
|---|---|---|---|
| sub_baseline_hybrid_k5 | BM25 + BGE-M3 v1 (title prepend) | 0.0 | 임베딩 prepend 부작용 |
| sub_v2_clean_k100 | BGE-M3 v2 clean | 0.0035 | 약간 회복 |
| sub_v2_quota_law15_court15 | v2 + source quota | 0.0 | quota top 너무 작음 (rank 36 첫 hit) |
| sub_v3_graph | v2 + graph 1-hop | 0.0094 | graph 효과 미미 |

**모든 LLM-free 접근법은 val F1 < 0.01**. 이는 이 task의 본질적 어려움을 보여줌:
- 사실관계 (1500자 영어 narrative) → 추상 법령 (독일어) abstraction gap
- 임베딩만으로는 못 건넘
- LLM이 사실관계 → 법률 개념 변환해야 함
