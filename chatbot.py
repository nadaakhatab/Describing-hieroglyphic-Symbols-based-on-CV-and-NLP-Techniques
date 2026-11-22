"""Single-file chatbot backend: settings, lookup, Groq route, and page loader.

The UI intentionally lives separately in templates/chat.html.  Vision and speech
remain project services, not part of the chatbot's request/response logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import quote
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.augmentation import build_context
from rag.retrieval import retrieve as rag_retrieve

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
STOP_WORDS = {
    "what", "does", "mean", "tell", "about", "this", "that", "with", "from",
    "have", "were", "they", "which", "your", "explain", "symbol", "glyph",
    "hieroglyph", "egyptian", "ancient",
}
SYSTEM_PROMPT = """You are an expert Ancient Egyptian Scribe and Egyptologist AI assistant for Kemet Studio.
Your mission is to provide authoritative, educational, and authentic answers about Ancient Egyptian civilization, hieroglyphs, language, history, religion, mythology, and grammar.

Key Principles:
1. GROUNDED ANSWERS: Use the supplied RAG evidence first. Cite factual claims with [S#]. If evidence is missing or conflicting, say so clearly and do not invent a source, translation, date, or Gardiner value.
2. ALWAYS INCLUDE AUTHENTIC HIEROGLYPHS (NO PLACEHOLDERS): Whenever referencing a hieroglyphic symbol, sign, or Egyptian word, ALWAYS include its genuine Unicode hieroglyph (e.g., 𓈖 for N35, 𓀀 for A1, 𓋹 for S34, 𓊹 for R8, 𓇳 for N5, 𓂋 for D21, 𓅓 for G17, etc.). NEVER output blanks, underscores `__`, or empty placeholders.
3. BEAUTIFULLY FORMATTED MARKDOWN: Format your responses with clear Markdown headings (##, ###), bullet lists, and conversational paragraphs. When providing a structured breakdown for a glyph, present it cleanly in an elegant Markdown table or clear bullet points with all cells completely filled.
4. DETAILED SIGN BREAKDOWN: For any glyph or symbol inquired about, provide:
   - Authentic Unicode hieroglyphic character (e.g. 𓈖)
   - Gardiner classification code (e.g. N35) & Gardiner Category (e.g. Category N: Sky, Earth, Water)
   - Phonetic transliteration (e.g. n) & Pronunciation guide
   - Sign type (uniliteral phonogram, ideogram, or determinative)
   - Cultural, religious, and historical symbolism in Ancient Egypt
5. LANGUAGE: Answer in the same language as the user's latest message. English question = English answer; Arabic question = Arabic answer. Do not translate the answer unless asked.
6. STYLE: Be warm and natural. For greetings, greet the user briefly. For research questions, use a short answer followed by 3-5 clear bullets. Avoid repeated headings and do not add a section called 'What the evidence tells us' unless it helps.
7. Always display relevant authentic hieroglyphic characters when they are known."""


@dataclass(frozen=True)
class Settings:
    """Server-side configuration; the API key never reaches the browser."""

    api_key: str = field(repr=False)
    model: str = "openai/gpt-oss-20b"
    data_path: Path = ROOT / "data" / "semantic_meaning.json"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key == "your_groq_api_key_here":
            key = ""
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        if not model:
            raise ValueError("GROQ_MODEL must not be empty.")
        path = Path(os.getenv("GLYPH_DATA_PATH", "data/semantic_meaning.json"))
        return cls(key, model, path if path.is_absolute() else ROOT / path)


class Turn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=10)
    glyph_codes: list[str] = Field(default_factory=list, max_length=6)
    image_context: str = Field(default="", max_length=4000)

    @field_validator("glyph_codes")
    @classmethod
    def validate_codes(cls, codes: list[str]) -> list[str]:
        if any(not code.strip() or len(code) > 40 for code in codes):
            raise ValueError("Each glyph identifier must contain 1–40 characters.")
        return [code.strip() for code in codes]


class ChatResponse(BaseModel):
    response: str
    context_keys: list[str] = Field(default_factory=list)
    knowledge_loaded: bool
    sources: list[dict] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    suggestions: list[str] = Field(default_factory=list)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def keywords(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[^\W\d_]+", text.casefold())
        if len(word) >= 4 and word not in STOP_WORDS
    }


def is_greeting(text: str) -> bool:
    """Return True for a short social greeting that needs no retrieval."""
    normalized = re.sub(r"[^a-z\u0600-\u06ff ]", "", text.casefold()).strip()
    return normalized in {"hi", "hello", "hey", "heyyy", "مرحبا", "اهلا", "أهلا", "السلام عليكم"}


def response_language(text: str) -> str:
    """Choose the answer language from the user's latest message."""
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    latin = len(re.findall(r"[a-z]", text.casefold()))
    return "Arabic" if arabic > latin else "English"


def web_search_allowed(question: str) -> bool:
    """Allow simple research searches, but keep risky requests local and safe."""
    risky = ("medical", "diagnose", "weapon", "harm", "steal", "loot", "forge", "دواء", "سلاح", "سرقة")
    return not any(word in question.casefold() for word in risky)


async def wikipedia_evidence(client: httpx.AsyncClient, question: str, language: str) -> list[dict]:
    """Find a short Wikipedia summary when the local evidence is not enough."""
    if not web_search_allowed(question):
        return []
    lang = "ar" if language == "Arabic" else "en"
    search_url = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        result = await client.get(search_url, params={
            "action": "query", "list": "search", "srsearch": question,
            "srlimit": 2, "format": "json", "utf8": 1,
        })
        result.raise_for_status()
        pages = result.json().get("query", {}).get("search", [])
        evidence = []
        for page in pages[:2]:
            title = page.get("title", "")
            if not title:
                continue
            summary = await client.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
            if summary.is_error:
                continue
            data = summary.json()
            text = data.get("extract", "").strip()
            if text:
                evidence.append({
                    "id": f"wikipedia-{lang}-{title}", "text": text,
                    "source": f"Wikipedia ({lang})", "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "similarity": 0.55,
                })
        return evidence
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def follow_up_suggestions(question: str, language: str) -> list[str]:
    """Offer two friendly next questions without making the answer noisy."""
    if language == "Arabic":
        return ["هل تريد معرفة الفترة التاريخية المرتبطة به؟", "هل تريد أن أشرح الرموز أو الشخصيات المرتبطة به؟"]
    return ["Would you like to know its historical period?", "Would you like me to explain the related signs or figures?"]


class KnowledgeBase:
    """Validated local descriptions and conservative code-aware retrieval."""

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
        ranked: list[tuple[int, str]] = []
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
        if selected or codes:
            return dict(list(selected.items())[:6])
        for _, key in sorted(ranked, key=lambda item: (-item[0], item[1])):
            if len(selected) >= 6:
                break
            selected.setdefault(key, self.records[key][:1800])
        return dict(list(selected.items())[:6])


class ChatError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message
        super().__init__(message)


class ChatService:
    def __init__(self, settings: Settings, knowledge: KnowledgeBase, client: httpx.AsyncClient):
        self.settings, self.knowledge, self.client = settings, knowledge, client

    async def reply(self, request: ChatRequest) -> ChatResponse:
        if not self.settings.api_key:
            raise ChatError(503, "Groq is not configured. Add GROQ_API_KEY to the project's .env file, then restart the server.")
        if is_greeting(request.message):
            greeting = "أهلًا! اسأليني عن الهيروغليفية أو تاريخ مصر القديمة." if response_language(request.message) == "Arabic" else "Hello! Ask me about hieroglyphs or Ancient Egyptian history."
            return ChatResponse(response=greeting, knowledge_loaded=bool(self.knowledge.records))
        prior = next((turn.content for turn in reversed(request.history) if turn.role == "user"), "")
        context = self.knowledge.retrieve(request.message, request.glyph_codes)
        if not context and prior:
            context = self.knowledge.retrieve(prior, request.glyph_codes)
        # Search the local knowledge base before asking Groq to write an answer.
        rag_records = rag_retrieve(request.message, request.glyph_codes, top_k=6)
        language = response_language(request.message)
        if not rag_records or max(record.get("similarity", 0) for record in rag_records) < 0.45:
            rag_records.extend(await wikipedia_evidence(self.client, request.message, language))
        rag_packet = build_context(request.message, rag_records)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(turn.model_dump() for turn in request.history)
        messages.append({"role": "user", "content": json.dumps({
            "question": request.message,
            "supplied_glyph_codes": request.glyph_codes,
            "image_context": request.image_context,
            "local_descriptions": context,
            "rag_evidence": rag_packet["context"],
            "rag_instructions": rag_packet["instructions"],
            "rag_citations": rag_packet["citations"],
            "answer_language": language,
            "web_search_note": "Web summaries may be used for simple background questions. Cite them as Wikipedia and keep uncertainty clear.",
        }, ensure_ascii=False)})
        payload = {
            "model": self.settings.model, "messages": messages,
            "temperature": 0.2, "max_completion_tokens": 1600,
        }
        if self.settings.model.startswith("openai/gpt-oss-"):
            payload.update(reasoning_effort="low", reasoning_format="hidden")
        # Send the question, conversation history, and grounded evidence together.
        try:
            result = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise ChatError(504, "The AI service timed out. Please try again.") from exc
        except httpx.RequestError as exc:
            raise ChatError(503, "The AI service is temporarily unreachable.") from exc
        if result.status_code == 429:
            raise ChatError(429, "API usage limit reached. Wait before trying again.")
        if result.status_code in (401, 403):
            raise ChatError(503, "The server's AI credentials or permissions need attention.")
        if result.is_error:
            raise ChatError(502, "The AI service rejected the request. Check the server configuration.")
        try:
            choice = result.json()["choices"][0]
            answer = choice["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty answer")
            if choice.get("finish_reason") == "length":
                answer += "\n\n[Response limit reached; ask for a shorter explanation.]"
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ChatError(502, "The AI service returned an invalid response.") from exc
        # Return both the answer and the evidence details needed by the UI.
        return ChatResponse(
            response=answer.strip(),
            context_keys=list(context),
            knowledge_loaded=bool(self.knowledge.records) or bool(rag_records),
            sources=[{**item, "similarity": record.get("similarity", 0.0)} for item, record in zip(rag_packet["citations"], rag_records)],
            retrieval_confidence=rag_packet["confidence"],
            suggestions=follow_up_suggestions(request.message, language),
        )


def create_chat_router(
    settings: Settings | None = None,
    knowledge: KnowledgeBase | None = None,
) -> APIRouter:
    settings = settings or Settings.from_env()
    knowledge = knowledge if knowledge is not None else KnowledgeBase.load(settings.data_path)
    router = APIRouter(tags=["Chatbot"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            try:
                return await ChatService(settings, knowledge, client).reply(request)
            except ChatError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.message) from exc

    return router
