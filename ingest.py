"""
PDF 문서를 파싱해 청크로 분할하고, 임베딩 후 ChromaDB에 저장.

동음이의어/다의어 대응 전략:
- 넉넉한 overlap으로 단어 주변 문맥이 임베딩에 반영되도록 함
- 청크에 문서 제목 + 페이지 메타데이터를 prefix로 붙여 문맥 강화
- multilingual-e5-large: "passage: " prefix 필수 (E5 포맷 규칙)
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


def load_pdf(pdf_path: Path) -> list[dict]:
    """PDF에서 페이지별 텍스트와 메타데이터를 추출."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = _clean_text(text)
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num,
                "source": pdf_path.name,
            })
    doc.close()
    return pages


def _clean_text(text: str) -> str:
    """불필요한 공백/특수문자 정리. 한국어 줄바꿈 처리 포함."""
    # 하이픈으로 끊긴 단어 복원 (영어 PDF에서 자주 발생)
    text = re.sub(r"-\n(\w)", r"\1", text)
    # 문장 중간 줄바꿈 제거 (한국어는 줄바꿈이 단어 구분자가 아님)
    text = re.sub(r"(?<![.!?。？！])\n(?![•\-\d\n])", " ", text)
    # 연속 공백 정리
    text = re.sub(r" {2,}", " ", text)
    # 연속 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """한국어/영어 혼합 문장 분리."""
    pattern = r"(?<=[.!?。？！])\s+"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    문장 단위로 청크 생성. CHUNK_SIZE 단어 이내로 묶고,
    CHUNK_OVERLAP 단어만큼 이전 청크와 겹침 (문맥 연속성 보장).
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

        current_sentences: list[str] = []
        current_word_count = 0

        for sent in sentences:
            word_count = len(sent.split())
            current_sentences.append(sent)
            current_word_count += word_count

            if current_word_count >= config.CHUNK_SIZE:
                chunk_text = " ".join(current_sentences)
                chunks.append(_make_chunk(chunk_text, page, chunk_id))
                chunk_id += 1

                # overlap: 뒤쪽 문장들을 다음 청크의 시작으로 이월
                overlap_words = 0
                overlap_sentences = []
                for s in reversed(current_sentences):
                    overlap_words += len(s.split())
                    overlap_sentences.insert(0, s)
                    if overlap_words >= config.CHUNK_OVERLAP:
                        break

                current_sentences = overlap_sentences
                current_word_count = sum(len(s.split()) for s in current_sentences)

        # 페이지 마지막 남은 문장들
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(_make_chunk(chunk_text, page, chunk_id))
            chunk_id += 1

    return chunks


def _make_chunk(text: str, page: dict, chunk_id: int) -> dict:
    """청크 딕셔너리 생성. 임베딩용 contextual_text에 문서/페이지 prefix 추가."""
    # 동음이의어 대응: 문서명 + 페이지를 prefix로 붙여 임베딩 문맥 강화
    contextual_text = (
        f"문서: {page['source']} | 페이지: {page['page']}\n{text}"
    )
    uid = hashlib.md5(f"{page['source']}_{chunk_id}".encode()).hexdigest()
    return {
        "id": uid,
        "text": text,
        "contextual_text": contextual_text,
        "source": page["source"],
        "page": page["page"],
        "chunk_index": chunk_id,
    }


class Embedder:
    def __init__(self):
        print(f"[Embedder] 모델 로드 중: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # E5 계열 모델은 passage에 "passage: " prefix 필수
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        # E5 계열 모델은 query에 "query: " prefix 필수
        return self.model.encode(
            f"query: {query}", normalize_embeddings=True
        ).tolist()


def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def ingest_pdfs(pdf_dir: Path = config.PDF_DIR) -> None:
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[Ingest] {pdf_dir}에 PDF 파일이 없습니다.")
        return

    print(f"[Ingest] PDF {len(pdf_files)}개 발견")

    embedder = Embedder()
    _, collection = get_chroma_collection()

    # 이미 인덱싱된 파일 추적
    existing_sources = set()
    if collection.count() > 0:
        results = collection.get(include=["metadatas"])
        existing_sources = {m["source"] for m in results["metadatas"]}

    for pdf_path in pdf_files:
        if pdf_path.name in existing_sources:
            print(f"[Ingest] 건너뜀 (이미 인덱싱됨): {pdf_path.name}")
            continue

        print(f"[Ingest] 처리 중: {pdf_path.name}")
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages)
        print(f"  → 청크 {len(chunks)}개 생성")

        # 배치 단위로 임베딩 + 저장
        batch_size = 32
        for i in tqdm(range(0, len(chunks), batch_size), desc="  임베딩"):
            batch = chunks[i : i + batch_size]
            embeddings = embedder.embed_passages(
                [c["contextual_text"] for c in batch]
            )
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "source": c["source"],
                        "page": c["page"],
                        "chunk_index": c["chunk_index"],
                        "contextual_text": c["contextual_text"],
                    }
                    for c in batch
                ],
            )

        # BM25용 청크 텍스트 별도 저장 (retriever.py에서 로드)
        _save_bm25_corpus(chunks, pdf_path.name)
        print(f"  → 저장 완료")

    print(f"[Ingest] 전체 완료. 총 벡터 수: {collection.count()}")


def _save_bm25_corpus(chunks: list[dict], source_name: str) -> None:
    """BM25 검색용 텍스트 코퍼스를 JSON으로 저장."""
    corpus_path = Path(config.CHROMA_DB_PATH) / "bm25_corpus.json"
    existing = []
    if corpus_path.exists():
        with open(corpus_path) as f:
            existing = json.load(f)

    # 중복 제거
    existing_ids = {item["id"] for item in existing}
    new_entries = [
        {"id": c["id"], "text": c["text"], "source": c["source"], "page": c["page"]}
        for c in chunks
        if c["id"] not in existing_ids
    ]
    existing.extend(new_entries)

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ingest_pdfs()
