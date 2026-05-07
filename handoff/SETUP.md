# SETUP — 새 서버 환경 구축

## 0. 사전 점검

```bash
nvidia-smi  # GPU 보이는지 확인
python --version  # 3.10+ 권장
nvcc --version 2>/dev/null || echo "nvcc 없음 (괜찮음)"
df -h .  # 디스크 여유 확인 (최소 30GB 필요: 캐시 + 모델)
```

기대값: RTX 5090 32GB, Python 3.10-3.12, 디스크 50GB+.

---

## 1. Python 환경

**venv 또는 conda 권장** — 시스템 Python 더럽히지 마세요. 사용자가 친구 서버 빌려쓰는 거라 깨끗하게.

```bash
cd ~  # 또는 적절한 작업 디렉토리
python -m venv legal-ir-env
source legal-ir-env/bin/activate
python -m pip install -U pip wheel setuptools
```

---

## 2. 핵심 패키지 설치

```bash
# PyTorch (CUDA 13.x — 5090은 sm_90/sm_100 컴퓨트, 최신 PyTorch 필수)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Transformers + 의존성
pip install transformers accelerate

# AWQ 양자화 모델 로드 (둘 중 하나 또는 둘 다)
pip install autoawq          # 전통적 AWQ 로더
pip install gptqmodel        # 최신 transformers는 이걸 선호

# Embedding + retrieval
pip install sentence-transformers faiss-cpu bm25s pystemmer

# 데이터
pip install pandas polars pyarrow

# 노트북 (필요 시)
pip install jupyter ipykernel
```

> **CUDA 13.2 + 드라이버 595.79** 환경이라 PyTorch nightly 또는 CUDA 12.4 빌드가 호환됩니다. 만약 sm_90 (RTX 5090) 미지원 에러 뜨면 nightly로:
> ```bash
> pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
> ```

## 3. 설치 검증

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Transformers 정상 import
from transformers import AutoTokenizer, AutoModelForCausalLM
print("transformers OK")

# Embedding model
from sentence_transformers import SentenceTransformer
print("sentence-transformers OK")

# FAISS
import faiss
print(f"faiss: {faiss.__version__}")
```

전부 OK 나오면 셋업 완료.

---

## 4. 작업 디렉토리 구조

```bash
mkdir -p ~/legal-ir/{data,parquet,indexes,graph,submissions,expansions,notebooks}
cd ~/legal-ir
```

또는 사용자 선호 위치. 이후 모든 코드의 경로를 이 디렉토리 기준으로 통일.

---

## 5. HuggingFace 캐시 위치 확인

기본은 `~/.cache/huggingface/hub`. 디스크 여유 없으면:

```bash
export HF_HOME=/path/to/big/disk/hf_cache
```

`~/.bashrc`에 추가하면 영구 적용.

---

## 6. 다음 단계

→ `DATA_TRANSFER.md` 따라가기

## 잠재 트러블 (RTX 5090 sm_120 관련)

RTX 5090은 sm_120 (Blackwell). 일부 라이브러리가 아직 미지원일 수 있음:

| 라이브러리 | 상태 | Workaround |
|---|---|---|
| PyTorch 2.4+ | 지원 | OK |
| Transformers | OK | - |
| AWQ (autoawq) | sm_120 컴파일 안 되어있을 수 있음 | gptqmodel 사용 |
| FAISS-GPU | sm_120 미지원 가능 | faiss-cpu 사용 (검색 충분히 빠름) |
| vLLM | sm_120 부분 지원 | transformers로 대체 |

**전반적 권장**: vLLM/FAISS-GPU 같은 컴파일된 GPU 라이브러리는 피하고, transformers + faiss-cpu로 가는 게 안전. RTX 5090 32GB면 메모리 충분해서 vLLM 안 써도 추론 충분히 빠름 (배치 처리).
