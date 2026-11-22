"""Dependency-light hybrid retrieval for the Kemet Studio knowledge base."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

try:
    import fitz  # PyMuPDF; optional until the book index is needed.
except ImportError:
    fitz = None

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = ROOT / "data" / "semantic_meaning.json"
RAG_DATA = Path(__file__).resolve().parent / "data"
BOOK_DIR = RAG_DATA / "books"
STOP = {
    "what", "does", "mean", "tell", "about", "this", "that", "with", "from",
    "explain", "symbol", "glyph", "hieroglyph", "egyptian", "ancient", "who",
    "was", "were", "is", "are", "the", "and", "how", "why", "when", "where",
}


@dataclass
class Document:
    id: str
    text: str
    source: str
    url: str
    metadata: dict

    def as_dict(self) -> dict:
        return asdict(self)


def terms(text: str) -> set[str]:
    return {
        word.casefold()
        for word in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
        if len(word) > 2 and word.casefold() not in STOP
    }


def load_book_documents() -> list[Document]:
    """Read local books page by page when PyMuPDF is installed."""
    if fitz is None or not BOOK_DIR.exists():
        return []
    documents: list[Document] = []
    for path in sorted(BOOK_DIR.glob("*.pdf")):
        try:
            pdf = fitz.open(path)
            for page_number, page in enumerate(pdf, start=1):
                text = " ".join(page.get_text("text").split())
                if len(text) < 80:
                    continue
                documents.append(Document(
                    id=f"book-{path.stem}-{page_number}",
                    text=text,
                    source=path.stem,
                    url="",
                    metadata={
                        "source_type": "book_pdf",
                        "local_file": str(path.relative_to(RAG_DATA)),
                        "page": page_number,
                    },
                ))
            pdf.close()
        except Exception:
            continue
    return documents


@lru_cache(maxsize=1)
def load_documents() -> tuple[Document, ...]:
    # First, load the original 179-sign catalog used by the Symbol Explorer.
    docs: list[Document] = list(load_book_documents())
    if LOCAL_DATA.exists():
        raw = json.loads(LOCAL_DATA.read_text(encoding="utf-8-sig"))
        for key, value in raw.items():
            story = value.get("story", "") if isinstance(value, dict) else str(value)
            parts = key.split("-", 1)
            code, glyph = parts[0], parts[1] if len(parts) > 1 else ""
            docs.append(Document(f"local-{code}-{glyph}", story, "Kemet Studio local catalog", "", {"gardiner": code, "glyph": glyph, "kind": "sign_story"}))
    # JSONL files are manually curated; retrieval never scrapes a website.
    for filename in ("signs.jsonl", "lexicon.jsonl", "texts.jsonl", "history.jsonl", "artifacts.jsonl", "books.jsonl", "articles.jsonl"):
        # Each JSONL file adds one small, focused knowledge collection.
        path = RAG_DATA / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            item = json.loads(line)
            text = item.get("text") or " | ".join(
                f"{key}: {value}"
                for key, value in item.items()
                if key not in {"id", "source_url"} and value
            )
            docs.append(Document(
                str(item.get("id", f"{filename}-{len(docs)}")), text,
                item.get("source", filename),
                item.get("source_url", item.get("url", "")), item,
            ))
    manifest = RAG_DATA / "source_manifest.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            docs.append(Document(item["id"], item["text"], item["source"], item["url"], item.get("metadata", {})))
    return tuple(docs)


def retrieve(query: str, codes: list[str] | None = None, top_k: int = 6, filters: dict | None = None) -> list[dict]:
    # Exact Gardiner or Unicode matches get the strongest ranking signal.
    codes = codes or []
    q_terms = terms(query)
    wanted = {x.casefold().strip() for x in codes}
    docs = load_documents()
    scored: list[tuple[float, Document, float, float]] = []
    for doc in docs:
        if filters and any(
            str(doc.metadata.get(key, "")).casefold() != str(value).casefold()
            for key, value in filters.items()
        ):
            continue
        hay = f"{doc.id} {doc.text} {json.dumps(doc.metadata, ensure_ascii=False)}".casefold()
        exact = 0
        for code in wanted:
            if code and code in hay:
                exact += 10
        # Keyword overlap keeps this version fast and easy to inspect locally.
        overlap = len(q_terms & terms(hay))
        overlap_ratio = overlap / max(1, len(q_terms))
        identity = " ".join([
            doc.source,
            str(doc.metadata.get("topic", "")),
            str(doc.metadata.get("title", "")),
            str(doc.metadata.get("local_file", "")),
        ])
        identity_ratio = len(q_terms & terms(identity)) / max(1, len(q_terms))
        topic_match = identity_ratio > 0
        source_bonus = {
            "unicode consortium": 1.5,
            "thesaurus linguae aegyptiae": 1.4,
        }.get(doc.source.casefold(), 0.0)
        if not exact and overlap == 0:
            continue
        score = exact + overlap / max(1, math.sqrt(len(q_terms))) + source_bonus + (2 if topic_match else 0)
        if score > 0:
            scored.append((score, doc, overlap_ratio, identity_ratio))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    # Keep the first result from each source before adding useful extra matches.
    selected: list[tuple[float, Document, float, float]] = []
    used_sources: set[str] = set()
    for item in scored:
        source = item[1].source.casefold()
        if source not in used_sources and len(selected) < top_k:
            selected.append(item)
            used_sources.add(source)
    for item in scored:
        if item not in selected and len(selected) < top_k:
            selected.append(item)
    return [
        {
            **doc.as_dict(),
            "score": round(score, 4),
            "similarity": round(
                min(
                    0.98,
                    0.92 if score >= 10 else 0.20 + (0.45 * identity_ratio) + (0.45 * overlap_ratio),
                ),
                3,
            ),
        }
        for score, doc, overlap_ratio, identity_ratio in selected
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--code", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(retrieve(args.query, args.code), ensure_ascii=False, indent=2))

