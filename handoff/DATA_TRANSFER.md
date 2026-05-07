# DATA_TRANSFER — 이전 캐시 옮기기 또는 raw에서 재빌드

## 옵션 A: 사용자 드라이브에서 캐시 통째로 가져오기 (권장)

이전 환경에서 빌드된 산출물이 사용자 Google Drive `LLM Legal Agent/` 폴더에 있습니다.

### A.1 사용자가 압축해서 전달

사용자에게 요청:
> "Google Drive `LLM Legal Agent/` 폴더 통째로 zip해서 서버에 업로드해주세요"

또는 대용량이라 (~5GB 추정) 폴더별로:
- `parquet/` (734MB) — 필수
- `indexes/` (~5GB) — BGE-M3 임베딩 + FAISS, 필수
- `graph/` (수 MB) — 옵션 (재빌드 10분이면 됨)
- `submissions/` — 참고용

### A.2 서버에 풀기

```bash
cd ~/legal-ir
unzip ~/uploads/legal_ir_cache.zip -d .
# 결과: ~/legal-ir/{parquet,indexes,graph,submissions}
ls -lah parquet/ indexes/
```

검증:
```python
from pathlib import Path
import pandas as pd
import numpy as np
import faiss

P = Path('~/legal-ir').expanduser()
val = pd.read_parquet(P / 'parquet/val.parquet')
assert len(val) == 10, f"val 손상: {len(val)}"

emb = np.load(P / 'indexes/bge_m3_v2_clean_fp16.npy', mmap_mode='r')
print(f"임베딩 shape: {emb.shape}, dtype: {emb.dtype}")
assert emb.shape[1] == 1024

law_index = faiss.read_index(str(P / 'indexes/law_v2.faiss'))
print(f"law_index ntotal: {law_index.ntotal:,}")
assert law_index.ntotal == 175933
```

OK 나오면 옵션 A 완료. → `PIPELINE_STATE.md`로 진행.

---

## 옵션 B: Raw 데이터에서 재빌드 (캐시 옮기기 어려울 때)

### B.1 데이터 다운로드

사용자 드라이브 공유 링크: `1ry3KYvcwnkuePLK2L2092GJzRsnLOeya` (789MB zip)

```bash
pip install gdown
cd ~/legal-ir
gdown 1ry3KYvcwnkuePLK2L2092GJzRsnLOeya -O data.zip
unzip data.zip -d data
ls -lah data/
```

기대 파일:
- `train.csv` (1.9MB)
- `val.csv` (20KB)
- `test.csv` (56KB)
- `laws_de.csv` (70MB)
- `court_considerations.csv` (2.3GB)
- `sample_submission.csv`

### B.2 Parquet 변환 (~30초)

```python
from pathlib import Path
import pandas as pd
import polars as pl

DATA_DIR = Path('~/legal-ir/data').expanduser()
PARQUET_DIR = Path('~/legal-ir/parquet').expanduser()
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# 작은 파일들
for fn in ['train.csv', 'val.csv', 'test.csv', 'laws_de.csv', 'sample_submission.csv']:
    df = pd.read_csv(DATA_DIR / fn)
    df.to_parquet(PARQUET_DIR / fn.replace('.csv', '.parquet'), compression='zstd')
    print(f'{fn}: {len(df):,}')

# 큰 court_considerations는 polars로
court = pl.read_csv(DATA_DIR / 'court_considerations.csv')
court.write_parquet(PARQUET_DIR / 'court_considerations.parquet', compression='zstd')
print(f'court: {len(court):,}')
```

### B.3 Corpus 재구성 + BGE-M3 임베딩 (~30-60분, RTX 5090에선 더 빠를 듯)

`PIPELINE_STATE.md`의 "Corpus build" + "Embedding build" 섹션 따라가기. 코드 다 들어있음.

### B.4 BM25 인덱스 + Citation graph

마찬가지로 `PIPELINE_STATE.md` 참고. 각 ~5-15분.

---

## 옵션 C: 임베딩 새로 만들기 + 데이터만 복사

이전 임베딩이 max_seq=256으로 빌드돼서, 더 긴 컨텍스트 (e.g. 512, 1024) 시도하고 싶으면 임베딩만 재빌드.

```python
# parquet, graph는 옵션 A로 복사
# 임베딩만 재계산:
# → PIPELINE_STATE.md "Embedding rebuild" 섹션
```

RTX 5090이면 max_seq=512 풀 빌드도 ~30-40분으로 가능할 듯 (이전 96GB GPU에서 256으로 60분이었는데 5090은 sm_120 + 실리콘 신형이라 비슷하거나 더 빠를 가능성).

---

## 검증 체크리스트

옮긴 후 또는 빌드 후 이 검증 코드 돌려서 정상인지 확인:

```python
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import pickle

P = Path('~/legal-ir').expanduser()

# 데이터
val = pd.read_parquet(P / 'parquet/val.parquet')
test = pd.read_parquet(P / 'parquet/test.parquet')
laws = pd.read_parquet(P / 'parquet/laws_de.parquet')
print(f"val: {len(val)}, test: {len(test)}, laws: {len(laws):,}")

# 임베딩
emb = np.load(P / 'indexes/bge_m3_v2_clean_fp16.npy')
print(f"emb: {emb.shape}, {emb.dtype}")

# FAISS
law_idx = faiss.read_index(str(P / 'indexes/law_v2.faiss'))
court_idx = faiss.read_index(str(P / 'indexes/court_v2.faiss'))
print(f"law_idx: {law_idx.ntotal:,}, court_idx: {court_idx.ntotal:,}")

# Doc IDs
with open(P / 'indexes/law_doc_ids_v2.pkl', 'rb') as f:
    LAW_DOC_IDS = pickle.load(f)
with open(P / 'indexes/court_doc_ids_v2.pkl', 'rb') as f:
    COURT_DOC_IDS = pickle.load(f)
print(f"LAW_DOC_IDS: {len(LAW_DOC_IDS):,}, COURT_DOC_IDS: {len(COURT_DOC_IDS):,}")

# Graph (옵션)
graph_path = P / 'graph/court_to_laws_v1.pkl'
if graph_path.exists():
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    print(f"graph: {len(graph):,} 판례에 인용 정보 (총 corpus 1.98M 중)")

print("\n✅ 모든 검증 통과")
```
