from fastapi.testclient import TestClient
import base64

import app
from chatbot import KnowledgeBase, Settings, create_chat_router
from vision import checkpoint_class_names


def test_health_and_real_story_lookup():
    with TestClient(app.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["story_records"] == 179
    reader = app.HieroglyphStoryReader(app.STORIES)
    assert reader.get_story("A1") == app.STORIES["A1-𓀀"]["story"]
    assert reader.get_story("𓀀") == app.STORIES["A1-𓀀"]["story"]
    assert reader.get_story("A1-𓀀") == app.STORIES["A1-𓀀"]["story"]


def test_chat_route_reports_missing_server_key_without_exposing_data():
    isolated = app.FastAPI()
    isolated.include_router(create_chat_router(Settings(""), KnowledgeBase({})))
    with TestClient(isolated) as client:
        response = client.post("/chat", json={"message": "What does S34 mean?"})
    assert response.status_code == 503
    assert "Groq is not configured" in response.json()["detail"]


def test_checkpoint_names_and_structured_detection_contract(monkeypatch):
    names = checkpoint_class_names(app.ROOT / "models" / "best.pt")
    assert len(names) == 1422
    assert names[0] == "A1"
    assert names[1421] == "Z9"

    class FakeDetector:
        def detect(self, image_bytes, mime_type):
            assert image_bytes == b"image"
            assert mime_type == "image/png"
            return {"is_hieroglyph": True, "screening_confidence": 0.91, "annotated_image": "aW1hZ2U=", "warnings": [], "detections": [{"label": "Symbol 1", "class_id": 0, "code": "A1", "confidence": 0.88, "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "story": "A seated man.", "uncertain": False}]}

    monkeypatch.setattr(app, "detector", lambda: FakeDetector())
    with TestClient(app.app) as client:
        response = client.post("/detect", files={"file": ("glyph.png", b"image", "image/png")})
    assert response.status_code == 200
    assert response.json()["detections"][0]["code"] == "A1"


def test_speak_contract_uses_generated_audio(monkeypatch):
    audio = b"RIFFmock-wav-data"
    monkeypatch.setattr(app, "synthesize_wav", lambda text: audio)
    with TestClient(app.app) as client:
        response = client.post("/speak", json={"text": "A seated man."})
    assert response.status_code == 200
    assert base64.b64decode(response.json()["audio"]) == audio
