"""Phase 0C 하드닝 회귀 테스트 (보안 감사 2026-07-07 대응)."""
import pytest
from fastapi import HTTPException

import symposium.session as session


# --- 입력 제약: top_k / 문자열 길이 ---

def test_top_k_over_limit_rejected(client):
    r = client.post("/api/search", json={"query": "삼위일체", "author": "moltmann", "top_k": 100000})
    assert r.status_code == 422


def test_top_k_below_min_rejected(client):
    r = client.post("/api/search", json={"query": "x", "author": "moltmann", "top_k": 0})
    assert r.status_code == 422


def test_query_max_length_rejected(client):
    r = client.post("/api/search", json={"query": "x" * 3000, "author": "moltmann"})
    assert r.status_code == 422


def test_search_ok_within_limits(client):
    r = client.post("/api/search", json={"query": "삼위일체", "author": "moltmann", "top_k": 5})
    assert r.status_code == 200
    assert r.json()["results"][0]["title"] == "제목"


def test_symposium_start_too_many_theologians(client):
    r = client.post("/api/symposium/start",
                    json={"theologians": ["a", "b", "c", "d", "e", "f"]})
    assert r.status_code == 422


def test_message_max_length_rejected(client):
    r = client.post("/api/symposium/ask", json={"session_id": "x", "message": "y" * 5000})
    assert r.status_code == 422


# --- 대화형 API 문서 비활성 ---

def test_docs_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/redoc").status_code == 404


# --- 보안 응답 헤더 ---

def test_security_headers_present(client):
    r = client.get("/api/authors")
    assert "Content-Security-Policy" in r.headers
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]


# --- confession-text 경로 이탈 방어 ---

def test_confession_path_traversal_rejected(web_module):
    for bad in ("..", "../../pyproject.toml", "../../../etc/hosts"):
        with pytest.raises(HTTPException) as ei:
            web_module._confession_path(bad)
        assert ei.value.status_code == 400


def test_confession_path_valid_within_dir(web_module):
    p = web_module._confession_path("westminster.html")
    assert p.parent == web_module.CONFESSIONS_DIR


# --- 예외 원문 유출 차단 ---

def test_search_error_generic_detail(client, web_module, monkeypatch):
    def boom(query, author, chroma_dir, top_k=5):
        raise RuntimeError("Collection [secret_internal] does not exist at /internal/path")
    monkeypatch.setattr(web_module, "search", boom)
    r = client.post("/api/search", json={"query": "x", "author": "zzz"})
    assert r.status_code == 404
    assert "Collection" not in r.text
    assert "internal" not in r.text


# --- 세션 하드닝 ---

def test_session_id_high_entropy():
    s = session.create_session(["moltmann"])
    assert len(s.session_id) > 12  # 기존 uuid4().hex[:12] 대비 향상


def test_session_history_capped():
    s = session.create_session(["moltmann"])
    for i in range(session.MAX_HISTORY + 10):
        session.add_message(s.session_id, "user", f"msg{i}")
    assert len(s.history) == session.MAX_HISTORY


def test_session_max_count_evicts(monkeypatch):
    monkeypatch.setattr(session, "MAX_SESSIONS", 3)
    for _ in range(5):
        session.create_session(["moltmann"])
    assert len(session._sessions) <= 3


def test_session_ttl_expiry():
    s = session.create_session(["moltmann"])
    s.last_active -= session.SESSION_TTL_SECONDS + 10
    assert session.get_session(s.session_id) is None
