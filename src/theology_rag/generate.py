"""검색된 발췌 + 질문 → Claude API 호출 (프롬프트 캐싱 적용)."""
from __future__ import annotations

import anthropic

from theology_rag.config import CLAUDE_MODEL

SYSTEM_PROMPT = """당신은 기독교 조직신학 전문 연구 조수입니다. 사용자가 제시한 참고 자료(특정 신학자의 원문 또는 번역 발췌)에 근거해 신학적 질문에 답합니다.

원칙:
1. 답변은 제공된 참고 자료를 근거로 해야 합니다. 자료에 직접 언급되지 않은 내용을 말할 때는 \"자료에는 직접 언급되지 않음\"이라고 명시하세요.
2. 각 신학적 주장 뒤에 출처를 `[저작명, p.페이지]` 형식으로 표기하세요.
3. 한국어로 답변하되, 중요한 신학 용어는 원어(독일어/영어/라틴어)를 괄호로 병기하세요.
4. 참고 자료가 부족해 답할 수 없으면 솔직히 밝히고, 추가로 필요한 자료의 성격을 제안하세요.
5. 조직신학적 맥락(삼위일체론·기독론·종말론·성령론·창조론·구원론·교회론)을 명시해 주면 사용자가 후속 연구를 이어가기 쉽습니다.
6. 추측·해석·요약을 구분하세요.
"""


def ask(question: str, hits: list[dict], model: str = CLAUDE_MODEL, max_tokens: int = 4096) -> tuple[str, dict]:
    client = anthropic.Anthropic()

    context_parts = []
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        source = f"[{m.get('title', '?')}, p.{m.get('page', '?')}]"
        context_parts.append(f"### 발췌 {i} {source}\n{hit['text']}")
    context = "\n\n".join(context_parts) if context_parts else "(검색 결과 없음)"

    user_content = f"# 참고 자료\n\n{context}\n\n# 질문\n\n{question}"

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )

    answer = next((b.text for b in response.content if b.type == "text"), "")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }
    return answer, usage
