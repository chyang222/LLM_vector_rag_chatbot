"""
Hybrid retrieval: Dense(ChromaDB) + Sparse(BM25) → RRF fusion → Cross-encoder reranking.

동음이의어/다의어 대응 전략:
- Dense: 의미론적 유사도 (같은 의미의 다른 표현 포착)
- BM25: 키워드 정확 매칭 (엉뚱한 의미의 청크를 Dense가 잘못 끌어올릴 때 보정)
- RRF: 두 결과를 순위 기반으로 합산 (점수 스케일 차이 무관)
- Cross-encoder: 쿼리-청크를 직접 비교해 의미 불일치 청크 최종 필터링
"""

import json
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from ingest import Embedder, get_chroma_collection


class BM25Index:
    def __init__(self):
        self.corpus: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        corpus_path = Path(config.CHROMA_DB_PATH) / "bm25_corpus.json"
        if not corpus_path.exists():
            return
        with open(corpus_path, encoding="utf-8") as f:
            self.corpus = json.load(f)
        tokenized = [self._tokenize(item["text"]) for item in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> list[str]:
        """
        한국어/영어 혼합 토크나이징.
        프로덕션에서는 KoNLPy 형태소 분석기 권장 (설치: pip install konlpy).
        현재는 공백 + 구두점 분리로 처리.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s가-힣]", " ", text)
        return text.split()

    def search(self, query: str, top_k: int) -> list[dict]:
        if self.bm25 is None or not self.corpus:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score > 0:
                item = self.corpus[idx]
                results.append({**item, "bm25_score": float(score)})
        return results


def _rrf_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = config.RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion: 두 순위 목록을 1/(k+rank) 합산으로 결합.
    점수 스케일이 달라도 공정하게 합산 가능.
    """
    scores: dict[str, float] = {}
    all_items: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        uid = item["id"]
        scores[uid] = scores.get(uid, 0) + config.DENSE_WEIGHT * (1 / (k + rank))
        all_items[uid] = item

    for rank, item in enumerate(bm25_results, start=1):
        uid = item["id"]
        scores[uid] = scores.get(uid, 0) + config.BM25_WEIGHT * (1 / (k + rank))
        all_items[uid] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{**all_items[uid], "rrf_score": scores[uid]} for uid in sorted_ids]


class Reranker:
    def __init__(self):
        print(f"[Reranker] 모델 로드 중: {config.RERANKER_MODEL}")
        # BAAI/bge-reranker-v2-m3: 한국어+영어 지원 다국어 cross-encoder
        self.model = CrossEncoder(config.RERANKER_MODEL, max_length=512)

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(chunks, scores), key=lambda x: x[1], reverse=True
        )
        return [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in ranked[: config.RERANK_TOP_K]
        ]


class HybridRetriever:
    def __init__(self):
        self.embedder = Embedder()
        self.bm25_index = BM25Index()
        self.reranker = Reranker()
        _, self.collection = get_chroma_collection()

    def retrieve(self, queries: list[str]) -> list[dict]:
        """
        여러 쿼리(동음이의어 disambiguation 결과)에 대해 검색 후 합산.
        단일 쿼리일 경우 queries = [original_query].
        """
        all_dense: list[dict] = []
        all_bm25: list[dict] = []

        for query in queries:
            dense = self._dense_search(query)
            bm25 = self.bm25_index.search(query, config.TOP_K_BM25)
            all_dense.extend(dense)
            all_bm25.extend(bm25)

        # 중복 제거 (같은 id 중 rrf_score 높은 것 유지)
        fused = _rrf_fusion(all_dense, all_bm25)
        seen = {}
        for item in fused:
            uid = item["id"]
            if uid not in seen or item["rrf_score"] > seen[uid]["rrf_score"]:
                seen[uid] = item
        fused_unique = sorted(seen.values(), key=lambda x: x["rrf_score"], reverse=True)

        # 대표 쿼리(첫 번째)로 reranking
        reranked = self.reranker.rerank(queries[0], fused_unique[:20])
        return reranked

    def _dense_search(self, query: str) -> list[dict]:
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(config.TOP_K_DENSE, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "id": meta.get("chunk_index", doc[:20]),
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "dense_score": float(1 - dist),  # cosine distance → similarity
            })
        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """LLM에 넘길 컨텍스트 문자열 생성."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[출처 {i}] {chunk['source']} (p.{chunk['page']})\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)
