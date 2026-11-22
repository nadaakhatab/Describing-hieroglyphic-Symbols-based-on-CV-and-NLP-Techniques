"""Safe generation wrapper for Groq or another OpenAI-compatible provider."""
from __future__ import annotations

import re
from typing import Callable

try:
    from .augmentation import prompt_messages
except ImportError:  # notebook / direct script execution from rag/
    from augmentation import prompt_messages


UNSAFE = ("how to forge", "how to loot", "how to steal", "make a weapon", "harm a person")


def safety_check(question: str) -> tuple[bool, str]:
    # Block harmful requests before they reach the external AI provider.
    normalized = question.casefold().strip()
    for phrase in UNSAFE:
        if phrase in normalized:
            return False, "I can help with the historical or archaeological context, but not instructions for harm, theft, or forgery."
    if len(question) > 2000:
        return False, "Please shorten the question to 2,000 characters or fewer."
    return True, ""


async def generate(question: str, records: list[dict], chat_call: Callable | None = None) -> dict:
    # Safety comes first; generation starts only after the question is allowed.
    allowed, message = safety_check(question)
    if not allowed:
        return {"response": message, "citations": [], "blocked": True}
    if chat_call is None:
        return {"messages": prompt_messages(question, records), "blocked": False}
    result = await chat_call(prompt_messages(question, records))
    # Keep accidental unsupported citation claims visible to the caller.
    citations = sorted(set(re.findall(r"\[S\d+\]", result)))
    return {"response": result, "citations": citations, "blocked": False}

