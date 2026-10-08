"""Small local lookup layer for project hieroglyph descriptions."""

import json
import logging
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)
STOP_WORDS = {"what", "does", "mean", "tell", "about", "this", "that", "with", "from", "have", "were", "they", "which", "your", "explain", "symbol", "glyph", "hieroglyph", "egyptian", "ancient"}


def keywords(text: str) -> set[str]:
    return {word for word in re.findall(r"[^\W\d_]+", text.casefold()) if len(word) >= 4 and word not in STOP_WORDS}


class KnowledgeBase:
    def __init__(self, records: dict[str, str]):
        self.records = records
        self.words = {key: keywords(story) for key, story in records.items()}

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        if not path.exists():
            LOGGER.warning("Glyph data is absent; local descriptions are unavailable.")
            return cls({})
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("Glyph data must be an object keyed by Gardiner code-symbol.")
        records = {}
        for key, value in raw.items():
            story = value.get("story") if isinstance(value, dict) else value
            if not isinstance(key, str) or not isinstance(story, str) or not story.strip():
                raise ValueError("Every glyph record must contain a non-empty story string.")
            records[key] = story.strip()
        return cls(records)

    def retrieve(self, query: str, codes: list[str]) -> dict[str, str]:
        selected: dict[str, str] = {}
        wanted = {code.casefold() for code in codes}
        ranked = []
        for key, story in self.records.items():
            parts = key.split("-", 1)
            identifiers = {key.casefold(), *(part.casefold() for part in parts)}
            exact = bool(wanted & identifiers)
            for part in parts:
                if re.fullmatch(r"[A-Za-z]+\d+[A-Za-z]*", part):
                    exact |= bool(re.search(r"(?<!\w)" + re.escape(part) + r"(?!\w)", query, re.IGNORECASE))
                elif part and all(0x13000 <= ord(char) <= 0x1345F for char in part):
                    exact |= part in query
            if exact:
                selected[key] = story[:1800]
            overlap = keywords(query) & self.words[key]
            if overlap:
                ranked.append((len(overlap), key))
        # A user or detector-supplied identifier is more reliable than broad
        # keyword overlap. Do not add unrelated stories for known or unknown codes.
        if selected or codes:
            return dict(list(selected.items())[:6])
        for _, key in sorted(ranked, key=lambda item: (-item[0], item[1])):
            if len(selected) >= 6:
                break
            selected.setdefault(key, self.records[key][:1800])
        return dict(list(selected.items())[:6])
