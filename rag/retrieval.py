"""Hybrid lexical + semantic retrieval for the Kemet Studio knowledge base."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import os

try:
    import fitz  # PyMuPDF; optional until the book index is needed.
except ImportError:
    fitz = None

if os.getenv("RAG_EMBEDDINGS_ENABLED", "false").casefold() == "true":
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None
    try:
        import chromadb
    except ImportError:
        chromadb = None
else:
    np, SentenceTransformer, chromadb = None, None, None

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = ROOT / "data" / "semantic_meaning.json"
RAG_DATA = Path(__file__).resolve().parent / "data"
BOOK_DIR = RAG_DATA / "books"
VECTOR_DIR = RAG_DATA / ".chroma"
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDINGS_ENABLED = os.getenv("RAG_EMBEDDINGS_ENABLED", "false").casefold() == "true"
LOAD_BOOKS = os.getenv("RAG_LOAD_BOOKS", "true").casefold() == "true"
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
    if not LOAD_BOOKS or fitz is None or not BOOK_DIR.exists():
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


class LocalVectorStore:
    """Persistent Chroma vector store with a lightweight fallback."""
    def __init__(self, documents: tuple[Document, ...]):
        self.documents, self.model, self.matrix, self.collection = documents, None, None, None
        self.enabled = EMBEDDINGS_ENABLED and chromadb is not None and SentenceTransformer is not None

    def _load_or_build(self) -> bool:
        if not self.enabled:
            return False
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            client = chromadb.PersistentClient(path=str(VECTOR_DIR))
            self.collection = client.get_or_create_collection(
                name="kemet_documents",
                metadata={"hnsw:space": "cosine"},
            )
            ids = [doc.id for doc in self.documents]
            current = self.collection.count()
            if current != len(ids):
                self.collection.delete(ids=self.collection.get().get("ids", []))
                self.collection.add(
                    ids=ids,
                    documents=[f"{doc.source}\n{doc.text}" for doc in self.documents],
                    metadatas=[{"source": doc.source, **{k: str(v) for k, v in doc.metadata.items()}} for doc in self.documents],
                    embeddings=self.model.encode(
                        [f"{doc.source}\n{doc.text}" for doc in self.documents],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).tolist(),
                )
            return True
        except Exception:
            self.model, self.matrix, self.collection = None, None, None
            return False

    def search(self, query: str, limit: int = 12) -> dict[int, float]:
        if self.collection is None and not self._load_or_build():
            return {}
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        result = self.collection.query(query_embeddings=[vector.tolist()], n_results=limit, include=["distances"])
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        index_by_id = {doc.id: index for index, doc in enumerate(self.documents)}
        return {
            index_by_id[doc_id]: max(0.0, 1.0 - float(distance))
            for doc_id, distance in zip(ids, distances)
            if doc_id in index_by_id
        }


@lru_cache(maxsize=1)
def vector_store() -> LocalVectorStore:
    return LocalVectorStore(load_documents())


def retrieve(query: str, codes: list[str] | None = None, top_k: int = 6, filters: dict | None = None) -> list[dict]:
    # Semantic search catches paraphrases; lexical search preserves exact codes.
    codes = codes or []
    q_terms = terms(query)
    wanted = {x.casefold().strip() for x in codes}
    docs = load_documents()
    semantic_scores = vector_store().search(query, limit=max(12, top_k * 2))
    scored: list[tuple[float, Document, float, float]] = []
    for index, doc in enumerate(docs):
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
        semantic_similarity = semantic_scores.get(index, 0.0)
        if not exact and overlap == 0 and semantic_similarity < 0.30:
            continue
        lexical_score = overlap / max(1, math.sqrt(len(q_terms)))
        score = exact + (0.65 * semantic_similarity) + (0.35 * lexical_score) + source_bonus + (2 if topic_match else 0)
        if score > 0:
            scored.append((score, doc, overlap_ratio, identity_ratio, semantic_similarity))
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
                    0.92 if score >= 10 else max(semantic_similarity, 0.20 + (0.30 * identity_ratio) + (0.30 * overlap_ratio)),
                ),
                3,
            ),
        }
        for score, doc, overlap_ratio, identity_ratio, semantic_similarity in selected
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--code", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(retrieve(args.query, args.code), ensure_ascii=False, indent=2))

