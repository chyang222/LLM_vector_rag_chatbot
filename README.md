# Vector RAG 챗봇

PDF 문서를 대상으로 한국어/영어 혼합 질의응답을 지원하는 RAG(Retrieval-Augmented Generation) 챗봇입니다. **동음이의어·다의어**를 자동으로 감지·분기 처리하는 것이 핵심 기능입니다.

## 아키텍처

```
PDF → Chunking → Embedding → ChromaDB
                                  ↓
Query → [Disambiguation] → Hybrid Retrieval → Reranking → Claude → 답변
          (Claude)          Dense + BM25         Cross-Encoder
```

| 단계 | 역할 |
|------|------|
| **Disambiguation** | Claude가 쿼리 내 모호한 단어를 탐지하고 의미별로 쿼리를 분기 |
| **Dense Search** | `multilingual-e5-large` 임베딩 + ChromaDB 코사인 유사도 검색 |
| **BM25 Search** | 키워드 기반 희소 검색으로 Dense의 의미 오염 보정 |
| **RRF Fusion** | Reciprocal Rank Fusion으로 두 결과를 점수 스케일 무관하게 합산 |
| **Reranking** | `bge-reranker-v2-m3` Cross-Encoder로 최종 5개 청크 선별 |
| **Generation** | Claude가 컨텍스트 기반 답변 생성 및 출처 명시 |

### 동음이의어 처리 예시

```
입력: "배 수출량은?"
  ↓
[Disambiguation] "배" = 선박 / 과일 모호 감지
  → 쿼리 분기: ["선박 수출량", "과일 배 수출량", "배 수출량"]
  ↓
[Retrieval] 3개 쿼리 각각 Dense + BM25 검색 → RRF 합산
  ↓
[Reranking] Cross-Encoder로 최종 5개 선별
  ↓
[Claude] 컨텍스트 기반 최종 답변 생성
```

## 파일 구조

```
rag/
├── chatbot.py        # 메인 CLI 챗봇 진입점
├── config.py         # 모델·경로·파라미터 설정
├── ingest.py         # PDF 파싱·청킹·임베딩·ChromaDB 저장
├── retriever.py      # Hybrid 검색 (Dense + BM25 + RRF + Reranker)
├── query_handler.py  # 동음이의어 disambiguation 및 프롬프트 구성
├── requirements.txt  # 의존성 목록
├── data/             # PDF 파일 저장 위치
└── chroma_db/        # ChromaDB 벡터 인덱스 (자동 생성)
```

## 설치 및 실행

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-...

python chatbot.py
```

## 챗봇 명령어

| 명령어 | 설명 |
|--------|------|
| `/ingest` | `data/` 폴더의 PDF를 인덱싱 (중복 자동 건너뜀) |
| `/reset` | 대화 기록 초기화 |
| `/debug` | 검색 상세 정보(출처·점수) 토글 |
| `/quit` | 종료 |

## 주요 설정 (`config.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | 한국어+영어 혼합 임베딩 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 다국어 Cross-Encoder |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | 답변 생성 및 disambiguation |
| `CHUNK_SIZE` | 500 (단어) | 문맥 확보를 위해 넉넉하게 설정 |
| `CHUNK_OVERLAP` | 150 (단어) | 청크 경계 의미 단절 방지 |
| `RERANK_TOP_K` | 5 | 최종 컨텍스트로 사용할 청크 수 |
| `DENSE_WEIGHT` | 0.6 | RRF에서 Dense 검색 가중치 |
| `BM25_WEIGHT` | 0.4 | RRF에서 BM25 검색 가중치 |

## 기술 스택

- **PDF 파싱**: PyMuPDF (`fitz`)
- **임베딩**: `sentence-transformers` / `intfloat/multilingual-e5-large`
- **벡터 DB**: ChromaDB (로컬 영구 저장)
- **희소 검색**: `rank-bm25` (BM25Okapi)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (Cross-Encoder)
- **LLM**: Anthropic Claude API
