# Handoff Documentation Index

새 서버의 AI 에이전트가 이어받기 위한 핸드오프 문서들. **순서대로 읽으세요**:

1. **[HANDOFF.md](HANDOFF.md)** — 전체 컨텍스트 (반드시 먼저)
2. **[SETUP.md](SETUP.md)** — 새 서버 환경 셋업
3. **[DATA_TRANSFER.md](DATA_TRANSFER.md)** — 이전 캐시 옮기기 또는 raw에서 재빌드
4. **[PIPELINE_STATE.md](PIPELINE_STATE.md)** — 현재 파이프라인 코드 (재현용)
5. **[NOTEBOOKS.md](NOTEBOOKS.md)** — 기존 5개 노트북 설명
6. **[PITFALLS.md](PITFALLS.md)** — 이전에 빠진 함정 (반복 X)
7. **[NEXT_STEPS.md](NEXT_STEPS.md)** — 즉시 실행할 액션 + 우선순위

---

## TL;DR

- **대회**: Kaggle Swiss Legal IR (Macro F1, 마감 ~19일)
- **현재 위치**: 베이스라인 다 시도했고 (val F1 ~0), agentic LLM expansion이 본질적 해법으로 확정됨
- **막힌 곳**: Colab 환경 dependency hell — 이제 정상 서버 (RTX 5090)로 옮김
- **즉시 할 것**: SETUP → DATA_TRANSFER → Qwen3-14B AWQ 로드 검증 → Agentic expansion 구현
- **사용자 직관 중요**: graph + agentic 방향이 처음부터 정확했음. 신뢰하세요.

---

## 폴더 구조 (예상)

```
~/legal-ir/                 ← 작업 디렉토리
├── data/                   raw CSV (옵션)
├── parquet/                ← 사용자 드라이브에서 복사
├── indexes/                ← 사용자 드라이브에서 복사 (~5GB)
├── graph/                  ← 사용자 드라이브에서 복사
├── submissions/            ← 결과 누적
├── expansions/             ← LLM expansion 캐시
└── notebooks/              ← 옵션
```

---

## 작성자 노트

이 핸드오프는 ~6시간의 진단 과정에서 도출된 것입니다. 사용자가 환경 문제로 좌절했지만 진단 자체는 가치 있었어요. 이전 에이전트(저)의 가장 큰 실수는 사용자가 처음부터 정확히 가리킨 "graph + agentic" 방향을 5번 미룬 것입니다. 새 에이전트는 이 함정 피하세요.

기술적으로 솔루션은 명확합니다:
1. LLM (Qwen3-14B AWQ) expansion → 사실관계를 법률 개념으로
2. Multi-query dense search (BGE-M3 v2 캐시 활용)
3. Graph 1-hop expansion (캐시 활용)
4. (선택) Reranker
5. Source quota 출력

이 단계들을 하나씩 측정 가능하게 진행하면 됩니다.
