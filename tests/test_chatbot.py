import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chatbot import ChatError, ChatRequest, ChatService, KnowledgeBase, Settings, create_chat_router


@pytest.fixture
def knowledge():
    return KnowledgeBase({"A1-𓀀": "A seated man.", "S34-𓋹": "The ankh is associated with life."})


@pytest.mark.parametrize("query,codes,expected", [
    ("What is A1?", [], "A1-𓀀"), ("𓋹", [], "S34-𓋹"),
    ("اشرح الرمز", ["S34"], "S34-𓋹"), ("What is an ankh?", [], "S34-𓋹"),
])
def test_retrieval(knowledge, query, codes, expected):
    assert expected in knowledge.retrieve(query, codes)


def test_explicit_codes_do_not_add_unrelated_keyword_matches(knowledge):
    assert list(knowledge.retrieve("Tell me more about daily life", ["A1"])) == ["A1-𓀀"]
    assert knowledge.retrieve("What does ZZ999 mean?", ["ZZ999"]) == {}


def test_validation_and_missing_key(knowledge):
    with pytest.raises(ValidationError):
        ChatRequest(message=" ")
    with pytest.raises(ValidationError):
        ChatRequest(message="hello", glyph_codes=[""])

    async def run():
        async with httpx.AsyncClient() as client:
            return await ChatService(Settings(""), knowledge, client).reply(ChatRequest(message="A1"))

    with pytest.raises(ChatError) as error:
        asyncio.run(run())
    assert error.value.status == 503


def test_mocked_provider_and_route(knowledge):
    def handler(request):
        body = json.loads(request.content)
        final = json.loads(body["messages"][-1]["content"])
        assert final["local_descriptions"]["S34-𓋹"].startswith("The ankh")
        return httpx.Response(200, json={"choices": [{"message": {"content": "The ankh means life."}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ChatService(Settings("test-secret"), knowledge, client).reply(ChatRequest(message="What is S34?"))

    assert asyncio.run(run()).response == "The ankh means life."
    app = FastAPI()
    app.include_router(create_chat_router(Settings(""), knowledge))
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": "hello"}).status_code == 503
