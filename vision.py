"""Lazy, checkpoint-verified hieroglyph image detection."""

from __future__ import annotations

import base64
import io
import pickletools
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from chatbot import KnowledgeBase

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class VisionError(RuntimeError):
    pass


class ImageValidationError(VisionError):
    pass


@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    code: str
    confidence: float
    bbox: dict[str, int]
    story: str | None
    uncertain: bool


def checkpoint_class_names(weights_path: Path) -> dict[int, str]:
    """Read the model's serialized names map without unpickling executable objects."""
    try:
        data = zipfile.ZipFile(weights_path).read("best/data.pkl")
        output = io.StringIO()
        pickletools.dis(data, out=output)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise VisionError("The detector checkpoint metadata could not be read.") from exc
    listing = output.getvalue()
    marker = "BINUNICODE 'names'"
    start = listing.find(marker)
    if start < 0:
        raise VisionError("The detector checkpoint has no verified class-name mapping.")
    # Pickle writes large dictionaries in SETITEMS batches (the 1,422 names
    # are split at index 1,000), so stop at the next model attribute instead.
    end = listing.find("BINUNICODE 'end2end'", start)
    if end < 0:
        raise VisionError("The detector checkpoint class-name mapping is incomplete.")
    section = listing[start:end]
    pairs = re.findall(r"BININT(?:1|2|4)?\s+(\d+).*?BINUNICODE '([^']+)'", section, re.DOTALL)
    names = {int(index): name for index, name in pairs}
    if not names:
        raise VisionError("The detector checkpoint class-name mapping is empty.")
    return names


@lru_cache(maxsize=1)
def load_models(weights_path: str):
    try:
        import clip
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise VisionError("Detection dependencies are not installed. Install the CV requirements.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return torch, clip, device, clip_model, preprocess, YOLO(weights_path)


class HieroglyphDetector:
    def __init__(self, weights_path: Path, knowledge: KnowledgeBase):
        self.weights_path = weights_path
        self.knowledge = knowledge
        self.names = checkpoint_class_names(weights_path)

    def detect(self, image_bytes: bytes, mime_type: str) -> dict:
        if mime_type not in ALLOWED_IMAGE_TYPES:
            raise ImageValidationError("Use a JPG, PNG, or WebP image.")
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ImageValidationError("The image must be between 1 byte and 10 MB.")
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise VisionError("Image dependencies are not installed.") from exc
        try:
            original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ImageValidationError("The uploaded file is not a readable image.") from exc
        image = original.resize((640, 640))
        torch, clip, device, clip_model, preprocess, yolo = load_models(str(self.weights_path))
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize(["ancient Egyptian hieroglyphs", "other content"]).to(device)
        with torch.no_grad():
            scores = (clip_model.encode_image(image_input) @ clip_model.encode_text(text_input).T).softmax(dim=-1)
        gate_score = float(scores[0][0].item())
        if gate_score <= 0.5:
            return {"is_hieroglyph": False, "screening_confidence": gate_score, "detections": [], "annotated_image": None, "warnings": ["The image-screening model did not identify hieroglyphs."]}
        result = yolo.predict(np.array(image), imgsz=640, verbose=False)[0]
        canvas = np.array(image)
        detections: list[Detection] = []
        boxes = result.boxes if result.boxes is not None else []
        for index, box in enumerate(boxes, start=1):
            class_id = int(box.cls.item())
            code = self.names.get(class_id)
            if code is None:
                continue  # Never create a label from an unknown numeric ID.
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
            story_matches = self.knowledge.retrieve("", [code])
            story = next(iter(story_matches.values()), None)
            item = Detection(f"Symbol {index}", class_id, code, confidence, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, story, confidence < 0.45)
            detections.append(item)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 102, 204), 2)
            cv2.putText(canvas, f"{item.label}: {code} {confidence:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 70, 140), 2, cv2.LINE_AA)
        encoded_ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not encoded_ok:
            raise VisionError("The annotated image could not be created.")
        warnings = []
        if not detections:
            warnings.append("No detector boxes passed the model output stage.")
        if any(item.uncertain for item in detections):
            warnings.append("Some detections are uncertain; check the highlighted labels.")
        if any(item.story is None for item in detections):
            warnings.append("Some detected codes have no local description.")
        return {"is_hieroglyph": True, "screening_confidence": gate_score, "detections": [item.__dict__ for item in detections], "annotated_image": base64.b64encode(encoded.tobytes()).decode("ascii"), "warnings": warnings}
