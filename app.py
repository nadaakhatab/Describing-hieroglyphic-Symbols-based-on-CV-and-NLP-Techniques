"""FastAPI backend for hieroglyph detection, stories, speech, and chat.

Heavy CV and speech dependencies load only when their endpoint is called. This
keeps the local chatbot available even on a machine without GPU-model packages.
"""

from __future__ import annotations

import base64
import asyncio
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from chatbot import KnowledgeBase, SpeakRequest, create_chat_router
from speech import SpeechError, synthesize_wav
from vision import HieroglyphDetector, ImageValidationError, VisionError

ROOT = Path(__file__).resolve().parent
STORY_PATH = ROOT / "data" / "Semantic meaning.json"
CHAT_PAGE = ROOT / "templates" / "chat.html"

app = FastAPI(title="Hieroglyph Assistant")


def load_stories() -> dict[str, dict[str, str]]:
    """Load the committed local descriptions and fail clearly if malformed."""
    try:
        raw = json.loads(STORY_PATH.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Story data is missing: {STORY_PATH.name}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("Story data must be a JSON object.")
    return raw


STORIES = load_stories()
KNOWLEDGE = KnowledgeBase.load(STORY_PATH)
app.include_router(create_chat_router(knowledge=KNOWLEDGE))


class QueryRequest(BaseModel):
    query: str


class HieroglyphStoryReader:
    def __init__(self, json_data: dict[str, dict[str, str]], lang_code: str = "a"):
        self.data = json_data
        self.lang_code = lang_code

    def get_story(self, query: str) -> str | None:
        normalized = query.strip()
        for key, value in self.data.items():
            gardiner, symbol = key.split("-", 1)
            if normalized in {key, gardiner, symbol}:
                story = value.get("story") if isinstance(value, dict) else value
                return story if isinstance(story, str) else None
        return None

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]", "", text)).strip()

    def speak_story(self, query: str) -> tuple[str | None, str]:
        story = self.get_story(query)
        if not story:
            return None, ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(synthesize_wav(story))
                return temp_file.name, self.clean_text(story)
        except SpeechError as exc:
            raise RuntimeError(str(exc)) from exc


@lru_cache(maxsize=1)
def detector() -> HieroglyphDetector:
    return HieroglyphDetector(ROOT / "models" / "best.pt", KNOWLEDGE)


@app.get("/", include_in_schema=False)
async def demo() -> HTMLResponse:
    return HTMLResponse(CHAT_PAGE.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "API is running", "story_records": len(STORIES)}


@app.post("/detect")
async def detect_hieroglyphs(file: UploadFile = File(...)):
    try:
        result = await asyncio.wait_for(asyncio.to_thread(detector().detect, await file.read(), file.content_type or ""), timeout=90)
        return JSONResponse(content=result)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Detection timed out. Try a smaller image.") from exc
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VisionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not process the image.") from exc


@app.post("/get-story-audio")
async def get_hieroglyph_story_audio(request: QueryRequest, background_tasks: BackgroundTasks):
    try:
        audio_path, transcription = HieroglyphStoryReader(STORIES).speak_story(request.query)
        if not audio_path:
            raise HTTPException(status_code=404, detail="No story for this hieroglyph symbol.")
        audio_base64 = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
        try:
            from googletrans import Translator
            translated_text = Translator().translate(transcription, src="en", dest="ar").text
        except Exception:
            translated_text = "Translation is currently unavailable."
        background_tasks.add_task(os.unlink, audio_path)
        return JSONResponse(content={"audio": audio_base64, "transcription": transcription, "translated_text": translated_text})
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/speak")
async def speak_chat_answer(request: SpeakRequest):
    try:
        audio = await asyncio.wait_for(asyncio.to_thread(synthesize_wav, request.text), timeout=90)
        return JSONResponse(content={"audio": base64.b64encode(audio).decode("ascii")})
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Speech generation timed out. Try a shorter answer.") from exc
    except SpeechError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
