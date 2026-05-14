from __future__ import annotations

from typing import Any

from loguru import logger

# Conservative token estimate: 1 token ≈ 3 chars.  Gemini responses contain
# lots of whitespace and short words that inflate char count vs tokens, so we
# use 3 chars/token to avoid underestimating and overflowing Gemma's context.
_CHARS_PER_TOKEN = 3


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _message_tokens(msg: dict) -> int:
    return _estimate_tokens(str(msg.get("content") or ""))


def total_tokens(messages: list[dict]) -> int:
    return sum(_message_tokens(m) for m in messages)


def add_message(messages: list[dict], new_msg: dict) -> list[dict]:
    """Return a new list with new_msg appended."""
    return messages + [new_msg]


async def maybe_summarize(
    messages: list[dict],
    engine: Any,
    settings: Any,
) -> list[dict]:
    """If total tokens exceed max_context_tokens, summarise the oldest tool
    observations (role='tool') and replace them with a single compact message.

    The summary call uses the same engine but with a very short generation.
    On any error the original messages list is returned unchanged.
    """
    max_ctx = getattr(settings, "agent_max_context_tokens", 6000)
    if total_tokens(messages) <= max_ctx:
        return messages

    # Collect tool-role messages (observations) from the oldest half
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_indices:
        return messages  # nothing we can compress

    # Summarise the first half of tool messages
    to_compress = tool_indices[: max(1, len(tool_indices) // 2)]
    combined_text = "\n\n".join(
        str(messages[i].get("content") or "") for i in to_compress
    )

    try:
        result = await engine.generate(
            model=settings.default_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this in 3 sentences:\n\n{combined_text}",
                }
            ],
            temperature=0.0,
            max_tokens=getattr(settings, "agent_memory_summary_max_tokens", 256),
            stop=None,
            tools=None,
            response_format="text",
            json_schema=None,
        )
        summary_text = result.content.strip() or combined_text[:500]
    except Exception as exc:
        logger.warning("memory summarize failed: {}", exc)
        return messages

    # Replace compressed messages with a single summary tool message
    summary_msg = {"role": "tool", "name": "_memory_summary", "content": summary_text}
    kept_indices = set(range(len(messages))) - set(to_compress)
    new_messages = [messages[i] for i in sorted(kept_indices)]
    # Insert summary after the last system/user message before the first removed index
    insert_at = min(to_compress)
    new_messages.insert(insert_at, summary_msg)
    return new_messages
