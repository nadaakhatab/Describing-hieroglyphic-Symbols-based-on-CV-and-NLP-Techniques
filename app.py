"""FastAPI backend for hieroglyph detection, stories, speech, and chat.

Heavy CV and speech dependencies load only when their endpoint is called. This
keeps the local chatbot available even on a machine without GPU-model packages.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from chatbot import chat_demo_html, create_chat_router

ROOT = Path(__file__).resolve().parent
STORY_PATH = ROOT / "data" / "Semantic meaning.json"

app = FastAPI(title="Hieroglyph Assistant")
app.include_router(create_chat_router())


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
            import soundfile as sf
            from kokoro import KPipeline

            pipeline = KPipeline(lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M")
            audio_data = None
            for output in pipeline(self.clean_text(story), voice="af_heart"):
                if isinstance(output, tuple) and len(output) >= 3:
                    audio_data = output[2]
            if audio_data is None:
                raise RuntimeError("No audio data received from Kokoro.")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, 24000)
                return temp_file.name, self.clean_text(story)
        except ImportError as exc:
            raise RuntimeError("Speech dependencies are not installed.") from exc


@lru_cache(maxsize=1)
def load_vision_models():
    """Load CLIP and YOLO once, only when image detection is requested."""
    try:
        import clip
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Detection dependencies are not installed.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return torch, clip, device, model, preprocess, YOLO(str(ROOT / "models" / "best.pt"))


@app.get("/", include_in_schema=False)
async def demo() -> HTMLResponse:
    return HTMLResponse(chat_demo_html())


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "API is running", "story_records": len(STORIES)}


@app.post("/detect")
async def detect_hieroglyphs(file: UploadFile = File(...)):
    try:
        import cv2
        import numpy as np
        from PIL import Image

        torch, clip, device, clip_model, preprocess, yolo_model = load_vision_models()
        image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((640, 640))
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize(["ancient Egyptian hieroglyphs", "other content"]).to(device)
        with torch.no_grad():
            similarity = (clip_model.encode_image(image_input) @ clip_model.encode_text(text_input).T).softmax(dim=-1)
        if similarity[0][0].item() <= 0.5:
            return JSONResponse(content={"message": "No hieroglyph found"})
        plotted = yolo_model.predict(np.array(image), imgsz=640)[0].plot()
        output = Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpeg")
    except RuntimeError as exc:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
