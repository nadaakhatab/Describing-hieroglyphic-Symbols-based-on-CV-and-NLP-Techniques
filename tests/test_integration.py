from fastapi.testclient import TestClient

import app
from chatbot import KnowledgeBase, Settings, create_chat_router


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
