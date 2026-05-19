"""
메인 CLI 챗봇. 실행: python chatbot.py

명령어:
  /ingest   - data/ 폴더의 PDF를 인덱싱
  /reset    - 대화 기록 초기화
  /debug    - 마지막 검색 결과 상세 출력
  /quit     - 종료
"""

import os
import sys
import anthropic

import config
from ingest import ingest_pdfs
from retriever import HybridRetriever
from query_handler import disambiguate_query, build_generation_prompt

_client = anthropic.Anthropic()


def _print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _print_disambiguation_info(info: dict) -> None:
    if info.get("is_ambiguous") and info.get("ambiguous_words"):
        words = ", ".join(info["ambiguous_words"])
        print(f"\n[쿼리 분석] 모호한 단어 감지: '{words}'")
        print(f"  이유: {info.get('reasoning', '')}")
        print(f"  검색 쿼리 ({len(info['queries'])}개): {info['queries']}")


def _print_sources(chunks: list[dict]) -> None:
    print("\n[검색된 출처]")
    for i, chunk in enumerate(chunks, start=1):
        score = chunk.get("rerank_score", chunk.get("rrf_score", 0))
        print(f"  [{i}] {chunk['source']} p.{chunk['page']} (score: {score:.3f})")


def answer_query(
    query: str,
    retriever: HybridRetriever,
    conversation_history: list[dict],
    debug: bool = False,
) -> tuple[str, list[dict], dict]:
    """
    단일 쿼리에 대한 전체 RAG 파이프라인 실행.
    Returns: (answer, retrieved_chunks, disambiguation_info)
    """
    # 1. 동음이의어/다의어 disambiguation
    disambiguation_info = disambiguate_query(query, conversation_history)

    if debug:
        _print_disambiguation_info(disambiguation_info)

    # 2. Hybrid retrieval (disambiguation된 쿼리들로 검색)
    chunks = retriever.retrieve(disambiguation_info["queries"])

    if not chunks:
        return "관련 문서를 찾을 수 없습니다.", [], disambiguation_info

    # 3. 컨텍스트 구성
    context = retriever.format_context(chunks)

    # 4. Claude로 답변 생성
    system_prompt, messages = build_generation_prompt(
        query, context, conversation_history, disambiguation_info
    )

    response = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        system=system_prompt,
        messages=messages,
    )
    answer = response.content[0].text

    return answer, chunks, disambiguation_info


def run_chatbot() -> None:
    print("=" * 60)
    print("  Vector RAG 챗봇 (동음이의어/다의어 처리 포함)")
    print("=" * 60)
    print("명령어: /ingest | /reset | /debug | /quit")
    _print_separator()

    # 초기화
    retriever: HybridRetriever | None = None
    conversation_history: list[dict] = []
    last_chunks: list[dict] = []
    debug_mode = False

    def get_retriever() -> HybridRetriever:
        nonlocal retriever
        if retriever is None:
            print("[시스템] 검색 엔진 초기화 중...")
            retriever = HybridRetriever()
            print("[시스템] 준비 완료")
        return retriever

    while True:
        try:
            user_input = input("\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n종료합니다.")
            break

        if not user_input:
            continue

        # 명령어 처리
        if user_input == "/quit":
            print("종료합니다.")
            break

        elif user_input == "/ingest":
            print("[Ingest] PDF 인덱싱 시작...")
            ingest_pdfs()
            retriever = None  # retriever 재초기화 (BM25 코퍼스 갱신)
            print("[Ingest] 완료. 검색 엔진을 재초기화합니다.")
            continue

        elif user_input == "/reset":
            conversation_history.clear()
            last_chunks.clear()
            print("[시스템] 대화 기록을 초기화했습니다.")
            continue

        elif user_input == "/debug":
            debug_mode = not debug_mode
            status = "ON" if debug_mode else "OFF"
            print(f"[시스템] 디버그 모드: {status}")
            if debug_mode and last_chunks:
                _print_sources(last_chunks)
            continue

        elif user_input.startswith("/"):
            print(f"알 수 없는 명령어: {user_input}")
            continue

        # RAG 답변 생성
        try:
            r = get_retriever()
            answer, chunks, disambiguation_info = answer_query(
                user_input, r, conversation_history, debug=debug_mode
            )
            last_chunks = chunks

            _print_separator()
            print(f"\nAI: {answer}")

            if debug_mode:
                _print_sources(chunks)

            _print_separator()

            # 대화 히스토리 업데이트 (원본 쿼리와 답변만 저장)
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})

            # 히스토리 길이 제한 (최근 10턴)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

        except anthropic.APIError as e:
            print(f"[오류] Claude API 오류: {e}")
        except Exception as e:
            print(f"[오류] {type(e).__name__}: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[오류] ANTHROPIC_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)
    run_chatbot()
