from pathlib import Path

# Paths
PDF_DIR = Path("./data")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_docs"

# Models
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Korean + English 혼합 최적
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"          # 다국어 cross-encoder
CLAUDE_MODEL = "claude-sonnet-4-6"

# Chunking
CHUNK_SIZE = 500        # 단어 기준 (동음이의어 문맥 확보를 위해 넉넉하게)
CHUNK_OVERLAP = 150     # 청크 경계에서 의미 단절 방지

# Retrieval
TOP_K_DENSE = 15        # dense 검색 후보 수
TOP_K_BM25 = 15         # BM25 검색 후보 수
RERANK_TOP_K = 5        # reranking 후 최종 청크 수

# RRF (Reciprocal Rank Fusion) fusion 가중치
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RRF_K = 60              # RRF 상수 (표준값)

# Claude generation
MAX_TOKENS = 2048
TEMPERATURE = 0.3
