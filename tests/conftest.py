"""pytest 공용 픽스처. 무거운 모델/CLI 호출을 스텁으로 대체한다."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def web_module():
    import symposium.web as web
    return web


@pytest.fixture(autouse=True)
def _clear_sessions():
    import symposium.session as s
    s._sessions.clear()
    yield
    s._sessions.clear()


@pytest.fixture
def client(monkeypatch, web_module):
    web = web_module

    def fake_search(query, author, chroma_dir, top_k=5):
        return [{
            "id": "x", "text": "발췌 텍스트",
            "metadata": {"title": "제목", "page": 1, "year": 2000},
            "distance": 0.12,
        }]

    async def fake_call_claude(prompt):
        return "테스트 답변"

    monkeypatch.setattr(web, "search", fake_search)
    monkeypatch.setattr(web, "_call_claude", fake_call_claude)
    return TestClient(web.app)
