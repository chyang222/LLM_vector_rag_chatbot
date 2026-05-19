"""
동음이의어/다의어 disambiguation 핵심 모듈.

전략:
1. Claude가 쿼리 내 잠재적으로 모호한 단어를 탐지
2. 모호한 경우 → 의미별로 명시적 쿼리 여러 개 생성 (multi-query)
3. 모호하지 않은 경우 → 쿼리 문맥 강화(expansion)만 수행
4. 생성된 쿼리들을 retriever에 전달 → 각각 검색 후 결과 합산

예시:
  입력: "배가 아프다"
  출력: ["복통이 있다 배가 아프다", ...]  # 선박 의미 제거됨

  입력: "배 수출량"
  출력: ["선박 수출량", "과일 배 수출량"]  # 양쪽 의미 커버
"""

import json
import anthropic

import config

_client = anthropic.Anthropic()

_DISAMBIGUATION_PROMPT = """당신은 정보 검색 시스템의 쿼리 분석 전문가입니다.
사용자 쿼리에서 동음이의어(같은 발음, 다른 의미)나 다의어(하나의 단어가 여러 의미)를 탐지하고,
검색 정확도를 높이기 위해 쿼리를 처리합니다.

다음 쿼리를 분석하세요: "{query}"

대화 맥락 (있는 경우):
{context}

분석 지침:
1. 쿼리에 모호한 단어가 있는지 확인
2. 모호한 경우: 각 의미별로 명확한 쿼리를 생성 (최대 3개)
3. 모호하지 않은 경우: 쿼리의 의미를 보존하면서 검색에 유리하게 확장 (1-2개)
4. 대화 맥락이 있으면 맥락을 활용해 의미를 좁힘

반드시 아래 JSON 형식으로만 답하세요:
{{
  "is_ambiguous": true/false,
  "ambiguous_words": ["단어1", "단어2"],
  "queries": ["쿼리1", "쿼리2", ...],
  "reasoning": "간단한 이유 설명"
}}"""


def disambiguate_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    쿼리를 분석해 동음이의어/다의어 처리 후 검색용 쿼리 목록을 반환.

    Returns:
        {
            "is_ambiguous": bool,
            "ambiguous_words": list[str],
            "queries": list[str],  # retriever에 넘길 쿼리들
            "reasoning": str,
        }
    """
    # 대화 맥락 요약 (최근 2턴만 사용)
    context_str = _format_context(conversation_history)

    prompt = _DISAMBIGUATION_PROMPT.format(query=query, context=context_str)

    try:
        response = _client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=512,
            temperature=0.1,  # disambiguation은 일관성이 중요하므로 낮게
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # JSON 파싱 (마크다운 코드블록 제거)
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        # 원본 쿼리가 항상 포함되도록 보장
        if query not in result["queries"]:
            result["queries"].insert(0, query)

        return result

    except (json.JSONDecodeError, KeyError, anthropic.APIError) as e:
        # 파싱 실패 시 원본 쿼리 그대로 사용
        return {
            "is_ambiguous": False,
            "ambiguous_words": [],
            "queries": [query],
            "reasoning": f"분석 실패, 원본 쿼리 사용: {e}",
        }


def _format_context(history: list[dict] | None) -> str:
    if not history:
        return "없음"
    # 최근 2턴 (user + assistant 쌍)
    recent = history[-4:]
    lines = []
    for turn in recent:
        role = "사용자" if turn["role"] == "user" else "AI"
        lines.append(f"{role}: {turn['content'][:200]}")
    return "\n".join(lines)


def build_generation_prompt(
    query: str,
    context: str,
    conversation_history: list[dict],
    disambiguation_info: dict,
) -> list[dict]:
    """Claude 답변 생성용 메시지 목록 구성."""

    system_prompt = (
        "당신은 제공된 문서를 기반으로 정확하게 답변하는 전문 어시스턴트입니다.\n"
        "반드시 아래 규칙을 따르세요:\n"
        "1. 제공된 [출처] 내용만을 근거로 답변하세요\n"
        "2. 문서에 없는 내용은 '제공된 문서에서 찾을 수 없습니다'라고 명시하세요\n"
        "3. 답변 마지막에 사용한 출처를 '[출처 N]' 형식으로 표시하세요\n"
        "4. 한국어로 답변하되, 원문이 영어인 경우 핵심 용어는 영어 병기 가능"
    )

    # 모호한 단어가 있었다면 이를 언급
    disambiguation_note = ""
    if disambiguation_info.get("is_ambiguous") and disambiguation_info.get("ambiguous_words"):
        words = ", ".join(disambiguation_info["ambiguous_words"])
        disambiguation_note = f"\n\n[참고] 쿼리의 '{words}'이(가) 다의적으로 해석될 수 있어 여러 의미로 검색했습니다."

    user_message = (
        f"다음 문서 내용을 참고해 질문에 답해주세요.\n\n"
        f"=== 검색된 문서 내용 ===\n{context}\n"
        f"========================\n\n"
        f"질문: {query}{disambiguation_note}"
    )

    messages = list(conversation_history)  # 대화 히스토리 복사
    messages.append({"role": "user", "content": user_message})

    return system_prompt, messages
