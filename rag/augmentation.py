"""Turn retrieved records into a grounded, citation-friendly prompt."""
from __future__ import annotations

import json


def build_context(question: str, records: list[dict], max_chars: int = 9000) -> dict:
    # Turn search results into short evidence blocks with stable source tags.
    blocks: list[str] = []
    citations: list[dict] = []
    used = 0
    for index, record in enumerate(records, start=1):
        source_tag = f"S{index}"
        text = record.get("text", "").strip()
        block = f"[{source_tag}] {text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        citations.append({"tag": source_tag, "source": record.get("source", "Unknown"), "url": record.get("url", ""), "id": record.get("id", "")})
        used += len(block)
    # These rules help the language model stay close to the retrieved evidence.
    instructions = (
        "Answer the question using the evidence below. Cite factual claims with [S#]. "
        "If the evidence is insufficient, say so clearly; do not invent a Gardiner value, "
        "translation, date, or historical claim. Preserve uncertainty and distinguish a sign's "
        "shape, phonetic value, word meaning, and cultural interpretation. "
        "Answer in the user's language, in a natural helpful tone. If no records are supplied, "
        "say that the local knowledge base has no verified evidence instead of guessing."
    )
    confidence = max((float(r.get("score", 0)) for r in records), default=0.0)
    return {"question": question, "instructions": instructions, "context": "\n\n".join(blocks), "citations": citations,
            "retrieval_note": "Evidence found; verify interpretation against cited sources." if records else "No matching evidence found.",
            "confidence": round(min(1.0, confidence / 12), 3)}


def prompt_messages(question: str, records: list[dict]) -> list[dict]:
    packet = build_context(question, records)
    return [
        {"role": "system", "content": packet["instructions"]},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]

