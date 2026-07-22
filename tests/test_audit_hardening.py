"""정밀검수(2026-07-22) 대응 회귀 테스트 — 게이트/본문상한/세션소유자/RAG충실도/설정."""
import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret-abc-123"


def _mk_jwt(role="approved", exp_delta=3600, aud=None, sub="u1", alg="HS256"):
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    header = {"alg": alg, "typ": "JWT"}
    payload = {"role": role, "exp": int(time.time()) + exp_delta, "sub": sub}
    if aud is not None:
        payload["aud"] = aud
    signing = f"{b64(header)}.{b64(payload)}"
    sig = base64.urlsafe_b64encode(
        hmac.new(SECRET.encode(), signing.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return f"{signing}.{sig}"


@pytest.fixture
def gated_client(monkeypatch, web_module):
    web = web_module
    monkeypatch.setattr(web, "_GATE_SECRET", SECRET)
    monkeypatch.setattr(web, "search", lambda *a, **k: [
        {"id": "x", "text": "발췌", "metadata": {"title": "T", "page": 1, "year": 2000}, "distance": 0.1}
    ])

    async def fake_claude(prompt):
        return "답변"

    monkeypatch.setattr(web, "_call_claude", fake_claude)
    return TestClient(web.app, base_url="http://symposium.nt-apparatus.com")


# --- 게이트: deny-by-default (3a) ---

def test_gate_local_host_ungated(client):
    # 로컬 host(conftest base_url=localhost)는 무게이트
    assert client.get("/api/authors").status_code == 200


def test_gate_blocks_nonlocal_without_token(gated_client):
    r = gated_client.get("/api/authors", headers={"accept": "application/json"})
    assert r.status_code == 401


def test_gate_html_redirects_to_login(gated_client):
    r = gated_client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 302
    assert "nt-apparatus.com/login" in r.headers["location"]


def test_gate_valid_token_allows(gated_client):
    r = gated_client.get("/api/authors", headers={"accept": "application/json"},
                         cookies={"access_token": _mk_jwt()})
    assert r.status_code == 200


def test_gate_fail_closed_when_secret_unset(monkeypatch, web_module):
    monkeypatch.setattr(web_module, "_GATE_SECRET", "")
    c = TestClient(web_module.app, base_url="http://symposium.nt-apparatus.com")
    r = c.get("/api/authors", headers={"accept": "application/json"})
    assert r.status_code == 503


# --- JWT 검증 (3b) ---

def test_jwt_aud_mismatch_rejected(web_module):
    assert web_module._verify_alex_jwt(_mk_jwt(aud="other"), SECRET) is False


def test_jwt_aud_symposium_ok(web_module):
    assert web_module._verify_alex_jwt(_mk_jwt(aud="symposium"), SECRET) is True


def test_jwt_aud_absent_backward_compatible(web_module):
    assert web_module._verify_alex_jwt(_mk_jwt(), SECRET) is True


def test_jwt_alg_none_rejected(web_module):
    forged = "eyJhbGciOiJub25lIn0.eyJyb2xlIjoiYWRtaW4iLCJleHAiOjk5OTk5OTk5OTl9."
    assert web_module._verify_alex_jwt(forged, SECRET) is False


def test_jwt_expired_rejected(web_module):
    assert web_module._verify_alex_jwt(_mk_jwt(exp_delta=-10), SECRET) is False


def test_jwt_wrong_role_rejected(web_module):
    assert web_module._verify_alex_jwt(_mk_jwt(role="pending"), SECRET) is False


def test_jwt_bad_signature_rejected(web_module):
    tok = _mk_jwt()
    assert web_module._verify_alex_jwt(tok + "x", SECRET) is False


# --- 본문 크기 상한: chunked 우회 차단 (2d) ---

def test_body_limit_chunked_rejected(client):
    def gen():
        for _ in range(30):
            yield b"a" * 10000  # 300KB, Content-Length 없음(chunked)
    r = client.post("/api/search", content=gen(), headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_body_limit_content_length_rejected(client):
    r = client.post("/api/search", content=b"a" * 300000,
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_body_small_ok(client):
    r = client.post("/api/search", json={"query": "삼위일체", "author": "moltmann"})
    assert r.status_code == 200


# --- 세션 소유자 결합 (3c) ---

def test_session_owner_blocks_other_user(gated_client):
    a = _mk_jwt(sub="alice")
    b = _mk_jwt(sub="bob")
    start = gated_client.post("/api/symposium/start", json={"theologians": ["moltmann"]},
                              cookies={"access_token": a})
    assert start.status_code == 200
    sid = start.json()["session_id"]
    # 다른 사용자(bob)는 alice 세션에 접근 불가 → 403
    r = gated_client.post("/api/symposium/direct",
                          json={"session_id": sid, "message": "안녕", "target": "moltmann"},
                          cookies={"access_token": b})
    assert r.status_code == 403


def test_session_owner_same_user_ok(gated_client):
    a = _mk_jwt(sub="alice")
    start = gated_client.post("/api/symposium/start", json={"theologians": ["moltmann"]},
                              cookies={"access_token": a})
    sid = start.json()["session_id"]
    r = gated_client.post("/api/symposium/direct",
                          json={"session_id": sid, "message": "안녕", "target": "moltmann"},
                          cookies={"access_token": a})
    assert r.status_code == 200


def test_session_local_ownerless_accessible(client):
    # 로컬(무게이트)에서 만든 세션은 owner 없음 → 접근 제한 없음(하위호환)
    start = client.post("/api/symposium/start", json={"theologians": ["moltmann"]})
    sid = start.json()["session_id"]
    r = client.post("/api/symposium/direct",
                    json={"session_id": sid, "message": "안녕", "target": "moltmann"})
    assert r.status_code == 200


# --- RAG 충실도: 용어집 단어경계 (1c) ---

def test_glossary_no_substring_false_positive(web_module):
    terms = web_module._find_relevant_terms("walking on the lawn, a single person")
    engs = {t.get("english") for t in terms}
    assert "law" not in engs
    assert "sin" not in engs


def test_glossary_true_positive_english(web_module):
    engs = {t.get("english") for t in web_module._find_relevant_terms("sin and grace")}
    assert "sin" in engs and "grace" in engs


def test_glossary_korean_josa_recall(web_module):
    # '은혜와'(조사 결합)에서 '은혜'는 매칭되어야 함
    kors = {t.get("korean") for t in web_module._find_relevant_terms("하나님의 은혜와 사랑")}
    assert "은혜" in kors


# --- RAG 충실도: 근거부재/저관련 프롬프트 (1a/1b) ---

def test_no_evidence_note_on_empty_hits(web_module):
    p = web_module._build_context("예정론이란?", [])
    assert web_module._NO_EVIDENCE_NOTE in p


def test_low_relevance_note_on_high_distance(web_module):
    bad = [{"text": "무관", "metadata": {"title": "X", "page": 1}, "distance": 0.9}]
    assert web_module._LOW_RELEVANCE_NOTE in web_module._build_context("김치", bad)


def test_no_low_relevance_note_on_good_hits(web_module):
    good = [{"text": "발췌", "metadata": {"title": "T", "page": 1}, "distance": 0.2}]
    assert web_module._LOW_RELEVANCE_NOTE not in web_module._build_context("예정론", good)


def test_citation_instruction_forbids_fabrication(web_module):
    assert "지어내지" in web_module.SYSTEM_INSTRUCTION
    assert "지어내지" in web_module.THEOLOGIAN_SYSTEM_TEMPLATE


# --- 설정 단일화 (4a/4c) ---

def test_model_pinned_sonnet(web_module):
    from symposium.config import CLAUDE_MODEL
    assert CLAUDE_MODEL == "claude-sonnet-4-6"
    assert web_module.CLAUDE_MODEL == "claude-sonnet-4-6"


def test_version_single_source(web_module):
    from symposium import __version__
    assert web_module.app.version == __version__
