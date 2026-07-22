"""FastAPI 웹 서버 — 신학 문헌 RAG 플랫폼."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from symposium import __version__
from symposium.config import (
    CHROMA_DIR,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT,
    METADATA_DIR,
    RELEVANCE_SOFT_MAX,
)
from symposium.retrieve import search
from symposium.session import create_session, get_session, add_message

logger = logging.getLogger("symposium")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
CONFESSIONS_DIR = (PROJECT_ROOT / "data" / "raw" / "confessions").resolve()

# 요청 본문 크기 상한(256KB) — 대형 페이로드로 인한 메모리/프롬프트 증폭 방지
MAX_BODY_BYTES = 256 * 1024


class _BodySizeLimitMiddleware:
    """실제 수신 바이트를 세어 본문 상한을 강제하는 순수 ASGI 미들웨어(2d).

    Content-Length 헤더만 보는 방식은 Transfer-Encoding: chunked(헤더 부재) 요청에
    우회된다. 여기서는 http.request 메시지의 body 길이를 누적해 상한 초과 시 413을 보내고,
    정상 요청은 버퍼링한 메시지를 그대로 재생(replay)해 다운스트림에 전달한다.
    누적 상한이 있어 메모리는 max_bytes 로 유계.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        buffered: list[dict] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                resp = JSONResponse({"detail": "요청 본문이 너무 큽니다."}, status_code=413)
                return await resp(scope, receive, send)
            buffered.append(message)
            if not message.get("more_body", False):
                break

        idx = 0

        async def replay():
            nonlocal idx
            if idx < len(buffered):
                m = buffered[idx]
                idx += 1
                return m
            return await receive()

        return await self.app(scope, replay, send)

# 대화형 API 문서(/docs·/redoc·/openapi.json)는 비활성 — 스키마 정찰 표면 축소
app = FastAPI(title="Symposium", version=__version__,
              docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── 외부(도메인) 접근 인증 게이트 (0.15.0) ───────────────────────
# symposium.nt-apparatus.com(Cloudflare 터널 경유) 요청만 Alexandria JWT 쿠키(access_token,
# HS256, 동일 JWT_SECRET)를 요구한다. 로컬 직접 접속(Host=127.0.0.1/localhost)은 무게이트 —
# 종전 로컬 전용 사용성 보존. 검증은 서명+만료+role(approved/admin)만(stdlib, DB조회 없음;
# 탈퇴자 즉시차단은 토큰만료 7일에 의존 — 소규모 사용자 수용, 2026-07-22 사용자 승인).
import base64 as _b64
import hashlib as _hashlib
import hmac as _hmac
import os as _os
import time as _time

from fastapi.responses import RedirectResponse

# 무게이트로 허용하는 로컬(루프백) 호스트. 이 목록에 없는 모든 host 는 인증 게이트 대상
# (deny-by-default). 화이트리스트(게이트할 host 나열) 대신 이 방식을 쓰면, 향후 새 도메인·
# 리버스프록시·재노출 경로가 추가돼도 목록 누락으로 무게이트 개방되는 사고가 없다(3a).
# 서버는 127.0.0.1 전용 바인딩이라 실제로 게이트되는 것은 cloudflared 터널 경유
# (symposium.nt-apparatus.com) 요청뿐이다.
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# 발급측(Alexandria)이 aud 클레임을 붙이면 강제 검증. 없으면 하위호환으로 통과(3b).
_EXPECTED_AUD = "symposium"
_ALEX_LOGIN_URL = "https://nt-apparatus.com/login"
_GATE_SECRET = _os.environ.get("JWT_SECRET", "")


def _b64url_decode(seg: str) -> bytes:
    return _b64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _decode_alex_jwt(token: str, secret: str) -> dict | None:
    """HS256 JWT 검증 후 payload 반환(실패 시 None). 외부 의존성 없이 stdlib만 사용.

    검증: alg==HS256 고정(alg 혼동 차단) · HMAC 서명 · exp · role∈{approved,admin} ·
    aud(있으면 'symposium' 이어야 함; 없으면 하위호환 통과).
    """
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "HS256":
            return None
        expected = _hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(),
                             _hashlib.sha256).digest()
        if not _hmac.compare_digest(expected, _b64url_decode(sig_b64)):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < _time.time():
            return None
        if payload.get("role") not in ("approved", "admin"):
            return None
        aud = payload.get("aud")
        if aud is not None and not (
            aud == _EXPECTED_AUD or (isinstance(aud, list) and _EXPECTED_AUD in aud)
        ):
            return None
        return payload
    except Exception:
        return None


def _verify_alex_jwt(token: str, secret: str) -> bool:
    """bool 래퍼(하위호환)."""
    return _decode_alex_jwt(token, secret) is not None


@app.middleware("http")
async def _external_auth_gate(request: Request, call_next):
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in _LOCAL_HOSTS:
        # 비-로컬 host 는 전부 인증 요구(deny-by-default)
        if not _GATE_SECRET:
            # 게이트 미구성 상태로 외부 서빙 금지(fail-closed)
            return JSONResponse({"detail": "인증 게이트가 구성되지 않았습니다."}, status_code=503)
        token = request.cookies.get("access_token", "")
        payload = _decode_alex_jwt(token, _GATE_SECRET)
        if payload is None:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                return RedirectResponse(url=_ALEX_LOGIN_URL, status_code=302)
            return JSONResponse({"detail": "로그인이 필요합니다."}, status_code=401)
        # 세션 소유자 결합용 신원 스탬프(3c). 로컬 무게이트 경로에는 없음.
        request.scope["auth_sub"] = str(payload.get("sub") or payload.get("email") or "")
    return await call_next(request)


@app.middleware("http")
async def _security_middleware(request: Request, call_next):
    # 본문 크기 상한은 _BodySizeLimitMiddleware(순수 ASGI)가 실제 수신 바이트 기준으로 강제한다.
    response = await call_next(request)
    # 보안 응답 헤더(심층방어). 인라인 스크립트 없음 → script-src 'self'.
    # script-src 'self'(인라인 스크립트/이벤트핸들러 차단=핵심 XSS 방어)는 엄격 유지.
    # style-src 는 UI가 동적 인라인 style 속성(era 색상 등)을 쓰므로 'unsafe-inline' 허용
    # (스타일은 JS 실행 불가라 위험 낮음). Pretendard 폰트는 jsdelivr(→fastly 리다이렉트) 허용.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fastly.jsdelivr.net; "
        "font-src 'self' https://cdn.jsdelivr.net https://fastly.jsdelivr.net; "
        "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# 본문 크기 상한을 실제 수신 바이트 기준으로 강제(가장 바깥 미들웨어 — 마지막 등록=최외곽).
app.add_middleware(_BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)


# claude CLI 동시 실행 방지 — 순차 처리 큐
_claude_lock = asyncio.Lock()
# 락 대기자 상한 — 초과 시 429(전역 락 무한 적체·워커 아사 방지)
_MAX_CLAUDE_WAITERS = 8
_claude_waiters = 0

# 프롬프트 지시문은 prompts.py 단일 진실원에서 가져온다(web·cli 공유, 드리프트 방지).
from symposium.prompts import SYSTEM_INSTRUCTION, THEOLOGIAN_SYSTEM_TEMPLATE


# --- Models ---

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    author: str = Field("moltmann", max_length=100)
    top_k: int = Field(5, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    author: str = Field("moltmann", max_length=100)
    top_k: int = Field(5, ge=1, le=20)


class AuthorInfo(BaseModel):
    key: str
    name_ko: str
    work_count: int
    born: int = 0
    tradition: str = ""


class SymposiumStartRequest(BaseModel):
    theologians: list[str] = Field(..., max_length=5)
    topic: str = Field("", max_length=200)
    confession: str = Field("", max_length=200)  # 신앙고백서 파일명
    confession_name: str = Field("", max_length=200)  # 한국어 제목

class SymposiumAskRequest(BaseModel):
    session_id: str = Field(..., max_length=64)
    message: str = Field(..., max_length=4000)

class SymposiumDirectRequest(BaseModel):
    session_id: str = Field(..., max_length=64)
    message: str = Field(..., max_length=4000)
    target: str = Field(..., max_length=100)


# --- Glossary ---

@lru_cache(maxsize=1)
def _load_glossary() -> list[dict]:
    """Theology_export_word.json 로드 (서버 시작 시 1회)."""
    path = PROJECT_ROOT / "Theology_export_word.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_HANGUL_RE = re.compile(r"[가-힣]")


def _kor_present(term: str, text: str) -> bool:
    """한국어 용어가 단어 경계로 등장하는지 검사(조사/합성어 오탐 방지).

    선행 문자가 한글 음절이면 더 긴 단어의 접미(예: '죄'가 '속죄'에)이므로 제외한다.
    후행 한글은 허용한다 — 한국어는 조사가 붙는 게 정상('은혜와'의 '은혜')이라
    후행까지 경계로 요구하면 정탐을 잃는다. 1자 용어는 오탐 위험이 커 제외한다.
    """
    if len(term) < 2:
        return False
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            return False
        before = text[idx - 1] if idx > 0 else ""
        if not _HANGUL_RE.match(before or " "):
            return True
        start = idx + 1


def _find_relevant_terms(text: str, max_terms: int = 30) -> list[dict]:
    """텍스트에서 매칭되는 신학 용어를 추출. 영어·한국어 양방향 검색.

    영어는 단어 경계(\\b) 정규식으로 대소문자 무시 매칭 — 부분문자열 오탐 방지
    (law→lawn, sin→single 등). 한국어는 조사/합성어 경계 검사(_kor_present).
    """
    glossary = _load_glossary()
    if not glossary:
        return []
    matched = []
    for entry in glossary:
        eng = entry.get("english", "")
        kor = entry.get("korean", "")
        hit = False
        if len(eng) >= 3:
            if re.search(r"\b" + re.escape(eng) + r"\b", text, re.IGNORECASE):
                hit = True
        if not hit and kor:
            hit = _kor_present(kor, text)
        if hit:
            matched.append(entry)
            if len(matched) >= max_terms:
                break
    return matched


def _format_glossary_section(terms: list[dict]) -> str:
    """용어집을 프롬프트 텍스트로 포맷."""
    if not terms:
        return ""
    lines = ["# 용어집 (아래 번역을 따르세요)", ""]
    for t in terms:
        lines.append(f"- {t['english']} → {t['korean']}")
    lines.append("")
    return "\n".join(lines)


# --- Helpers ---

def _get_available_authors() -> list[AuthorInfo]:
    """메타데이터 YAML이 있는 저자 목록 반환 (시대순 정렬)."""
    import yaml

    authors = []
    for yml in sorted(METADATA_DIR.glob("*.yaml")):
        if yml.stem == "confessions":
            continue
        with open(yml, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        author_info = meta.get("author", {})
        works = meta.get("works", [])
        born = author_info.get("born", 0)
        authors.append(AuthorInfo(
            key=author_info.get("key", yml.stem),
            name_ko=author_info.get("name_ko", yml.stem),
            work_count=len(works),
            born=born if isinstance(born, int) else 0,
            tradition=author_info.get("tradition", ""),
        ))
    authors.sort(key=lambda a: a.born)
    return authors


# 검색 실패·빈 결과 시 프롬프트에 주입 — 근거 없는 인용 조작 방지(1b)
_NO_EVIDENCE_NOTE = (
    "이번 질문에는 참고할 발췌가 제공되지 않았습니다. "
    "당신의 신학적 이해로 답하되, 특정 저작·페이지 인용을 지어내지 마세요."
)
# 최상위 hit 거리가 소프트 임계를 넘을 때 주입 — 무관 발췌 강제 인용 방지(1a)
_LOW_RELEVANCE_NOTE = (
    "주의: 아래 발췌는 질문과 충분히 관련되지 않을 수 있습니다. "
    "실제로 근거가 되는 발췌만 인용하고, 무관하면 인용하지 마세요."
)


def _min_distance(hits: list[dict]) -> float | None:
    ds = [h.get("distance") for h in hits if h.get("distance") is not None]
    return min(ds) if ds else None


def _reference_parts(hits: list[dict], header: str) -> list[str]:
    """참고 자료 섹션 조립. 발췌 없으면 근거부재 안내, 저관련이면 주의 삽입."""
    if not hits:
        return [f"# {header}", "", _NO_EVIDENCE_NOTE, ""]
    parts = [f"# {header}", ""]
    md = _min_distance(hits)
    if md is not None and md > RELEVANCE_SOFT_MAX:
        parts.append(_LOW_RELEVANCE_NOTE)
        parts.append("")
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        source = f"[{m.get('title', '?')}, p.{m.get('page', '?')}]"
        parts.append(f"### 발췌 {i} {source}\n{hit['text']}\n")
    return parts


def _build_context(question: str, hits: list[dict]) -> str:
    """검색 결과를 claude CLI에 전달할 프롬프트로 조합."""
    hit_texts = " ".join(h["text"] for h in hits)
    terms = _find_relevant_terms(question + " " + hit_texts)
    glossary = _format_glossary_section(terms)

    parts = [SYSTEM_INSTRUCTION, ""]
    if glossary:
        parts.append(glossary)
    parts.extend(_reference_parts(hits, "참고 자료"))
    parts.append(f"# 질문\n\n{question}")
    return "\n".join(parts)


def _get_author_meta(author_key: str) -> dict:
    """YAML 메타데이터에서 저자 정보 반환."""
    import yaml
    yml = METADATA_DIR / f"{author_key}.yaml"
    if not yml.exists():
        return {"key": author_key, "name_ko": author_key, "tradition": ""}
    with open(yml, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    info = meta.get("author", {})
    return {
        "key": info.get("key", author_key),
        "name_ko": info.get("name_ko", author_key),
        "tradition": info.get("tradition", ""),
        "born": info.get("born", ""),
    }


def _build_theologian_prompt(
    author_meta: dict,
    question: str,
    hits: list[dict],
    history: list[dict],
    confession_hits: list[dict] | None = None,
    confession_name: str = "",
) -> str:
    """특정 신학자의 관점으로 답변할 프롬프트 구성."""
    system = THEOLOGIAN_SYSTEM_TEMPLATE.format(
        name_ko=author_meta["name_ko"],
        tradition=author_meta.get("tradition", ""),
    )
    hit_texts = " ".join(h["text"] for h in hits)
    terms = _find_relevant_terms(question + " " + hit_texts)
    glossary = _format_glossary_section(terms)

    parts = [system, ""]
    if glossary:
        parts.append(glossary)

    # 신앙고백서 토론 컨텍스트
    if confession_hits and confession_name:
        parts.extend([f"# 토론 대상: {confession_name}", ""])
        for i, hit in enumerate(confession_hits, 1):
            parts.append(f"### 고백서 발췌 {i}\n{hit['text']}\n")
        parts.append("위 신앙고백서의 내용에 대해 당신의 신학적 관점에서 논평하세요.\n")

    parts.extend(_reference_parts(hits, "참고 자료 (본인 저작에서 발췌)"))

    if history:
        parts.append("# 이전 대화\n")
        for entry in history:
            if entry["role"] == "user":
                parts.append(f"[질문자] {entry['text']}\n")
            else:
                parts.append(f"[{entry.get('name_ko', '?')}] {entry['text']}\n")

    parts.append(f"# 현재 질문\n\n{question}")
    return "\n".join(parts)


def _load_recommended_questions() -> dict:
    path = PROJECT_ROOT / "data" / "recommended_questions.json"
    if not path.exists():
        return {"topics": {}, "combos": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def _call_claude(prompt: str) -> str:
    """claude CLI를 subprocess로 호출하여 답변 생성."""
    global _claude_waiters
    # 락 대기자 상한: 전역 락이 이미 한도만큼 적체돼 있으면 즉시 429(무한 큐잉·워커 아사 방지)
    if _claude_waiters >= _MAX_CLAUDE_WAITERS:
        raise HTTPException(status_code=429, detail="서버가 혼잡합니다. 잠시 후 다시 시도하세요.")
    _claude_waiters += 1
    try:
        async with _claude_lock:
            # 보안: --tools "" 로 도구를 구조적으로 비활성화한다. 전역 ~/.claude/settings.json 이
            # defaultMode=bypassPermissions + allow=[Bash(*)/Edit(*)/...] 이므로, 이 플래그가 없으면
            # 사용자 입력(프롬프트 인젝션)이 호스트 셸 실행(RCE)으로 이어진다. 신학 답변은 순수 텍스트
            # 생성이라 도구가 불필요하다. (실증: 플래그 없으면 양성 프롬프트로도 파일 생성됨)
            proc = await asyncio.create_subprocess_exec(
                "claude", "--print", "--tools", "", "--model", CLAUDE_MODEL,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode("utf-8")),
                    timeout=CLAUDE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("claude 호출 타임아웃(%ss) — 프로세스 종료", CLAUDE_TIMEOUT)
                raise HTTPException(status_code=504, detail="응답 생성 시간 초과")
            finally:
                # 타임아웃·취소(클라이언트 disconnect) 등 어떤 종료 경로에서든 subprocess 가
                # 아직 살아있으면 회수한다. proc.kill()은 동기(시그널 전송)라 취소 중에도 확실히
                # 실행되어, 고아 프로세스가 백그라운드에서 구독 요금제를 계속 소모하는 것을 막는다.
                if proc.returncode is None:
                    proc.kill()
                    try:
                        await proc.wait()
                    except Exception:
                        pass
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")
                logger.error("claude CLI 오류(rc=%s): %s", proc.returncode, err[:500])
                raise HTTPException(status_code=502, detail="응답 생성에 실패했습니다.")
            return stdout.decode("utf-8").strip()
    finally:
        _claude_waiters -= 1


# --- Routes ---

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/masterplan")
async def masterplan():
    return FileResponse(str(STATIC_DIR / "masterplan.html"))


@app.get("/api/authors")
async def list_authors():
    return _get_available_authors()


@app.get("/api/confessions")
async def list_confessions():
    """신앙고백서 개별 문서를 year 기준으로 반환."""
    import yaml
    yml = METADATA_DIR / "confessions.yaml"
    if not yml.exists():
        return []
    with open(yml, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    works = meta.get("works", [])
    return [
        {
            "key": "confessions",
            "title": w.get("title", ""),
            "year": w.get("year", 0),
            "tradition": w.get("tradition", ""),
            "file": w.get("file", ""),
        }
        for w in sorted(works, key=lambda x: x.get("year", 0))
    ]


@app.post("/api/search")
async def api_search(req: SearchRequest):
    try:
        hits = search(req.query, req.author, CHROMA_DIR, top_k=req.top_k)
    except Exception:
        logger.exception("검색 실패 (author=%s)", req.author)
        raise HTTPException(status_code=404, detail="검색 결과를 찾을 수 없습니다.")
    return {
        "query": req.query,
        "author": req.author,
        "results": [
            {
                "title": h["metadata"].get("title", "?"),
                "page": h["metadata"].get("page", "?"),
                "year": h["metadata"].get("year"),
                "distance": round(h["distance"], 3),
                "text": h["text"],
            }
            for h in hits
        ],
    }


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    try:
        hits = search(req.question, req.author, CHROMA_DIR, top_k=req.top_k)
    except Exception:
        logger.exception("검색 실패 (author=%s)", req.author)
        raise HTTPException(status_code=404, detail="검색 결과를 찾을 수 없습니다.")
    if not hits:
        raise HTTPException(status_code=404, detail="검색 결과 없음")

    prompt = _build_context(req.question, hits)
    answer = await _call_claude(prompt)

    return {
        "question": req.question,
        "author": req.author,
        "answer": answer,
        "sources": [
            {
                "title": h["metadata"].get("title", "?"),
                "page": h["metadata"].get("page", "?"),
                "year": h["metadata"].get("year"),
                "distance": round(h["distance"], 3),
                "text": h["text"][:300],
            }
            for h in hits
        ],
    }


@app.get("/symposium")
async def symposium_page():
    return FileResponse(str(STATIC_DIR / "symposium.html"))


def _confession_path(filename: str) -> Path:
    """confessions 디렉터리 내부로 한정된 안전 경로 반환. 경로 이탈 시 400.

    프레임워크 라우팅(슬래시 미매칭)에만 의존하지 않고 코드에서 격리 경계를 강제한다:
    resolve() 후 부모가 정확히 CONFESSIONS_DIR 여야 한다('..'·심볼릭·인코딩 우회 차단).
    """
    candidate = (CONFESSIONS_DIR / filename).resolve()
    if candidate.parent != CONFESSIONS_DIR:
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다.")
    return candidate


@app.get("/api/confession-text/{filename}")
async def confession_text(filename: str):
    """신앙고백서 전문 반환. 한글 번역 캐시 우선, 긴 텍스트는 목차만."""
    import re
    from symposium.ingest import extract_text_file, clean_text

    raw_path = _confession_path(filename)
    if not raw_path.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")

    # 한글 번역 캐시 확인
    ko_stem = Path(filename).stem
    ko_path = CONFESSIONS_DIR / f"{ko_stem}.ko.txt"
    if ko_path.exists():
        content = ko_path.read_text(encoding="utf-8", errors="replace").strip()
        return {"filename": filename, "text": content, "mode": "full", "lang": "ko"}

    # 영문 원본 로드
    content = raw_path.read_text(encoding="utf-8", errors="replace")
    if filename.endswith(".html") or filename.endswith(".htm"):
        content = extract_text_file(raw_path)
    content = clean_text(content)

    # 5,000자 이하: 전문 (번역 필요 표시)
    if len(content) <= 5000:
        return {"filename": filename, "text": content, "mode": "full", "lang": "en"}

    # 5,000자 초과: 목차 추출
    lines = content.split("\n")
    toc = []
    heading_re = re.compile(r"^(CHAPTER|ARTICLE|Article|Chapter|Part|PART|QUESTION|Question|Q\.|Lord'?s Day|SECTION|Section|CANON|Canon|Head|HEAD)\b")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            continue
        if heading_re.match(stripped):
            # 다음 줄이 제목인지 확인 (짧고 본문이 아닌 줄)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) < 100 and not heading_re.match(next_line):
                    toc.append(f"{stripped} — {next_line}")
                    continue
            toc.append(stripped)

    if not toc:
        toc = [l.strip() for l in lines if l.strip() and len(l.strip()) < 80 and l.strip().isupper()][:30]

    toc_text = "\n".join(toc) if toc else "(목차를 추출할 수 없습니다)"
    return {"filename": filename, "text": toc_text, "mode": "toc", "lang": "en", "total_length": len(content)}


def _check_session_owner(session, request: Request) -> None:
    """외부 인증 사용자 간 세션 이어쓰기 방지(3c). 소유자가 있고 요청자 신원이 다르면 403.

    로컬 무게이트(신원 없음)·소유자 없는 세션은 통과(하위호환) — 로컬 사용성 보존.
    """
    sub = request.scope.get("auth_sub", "")
    if session.owner and sub and session.owner != sub:
        raise HTTPException(403, "이 세션에 접근할 권한이 없습니다.")


@app.post("/api/symposium/start")
async def symposium_start(req: SymposiumStartRequest, request: Request):
    if len(req.theologians) > 5:
        raise HTTPException(400, "최대 5명까지 선택 가능합니다.")
    if len(req.theologians) < 1:
        raise HTTPException(400, "최소 1명을 선택해야 합니다.")

    session = create_session(req.theologians, confession=req.confession,
                             confession_name=req.confession_name,
                             owner=request.scope.get("auth_sub", ""))
    rq = _load_recommended_questions()

    combo_questions = []
    keys_sorted = sorted(req.theologians)
    for i in range(len(keys_sorted)):
        for j in range(i + 1, len(keys_sorted)):
            combo_key = f"{keys_sorted[i]}+{keys_sorted[j]}"
            combo_questions.extend(rq.get("combos", {}).get(combo_key, []))

    topic_questions = rq.get("topics", {}).get(req.topic, [])
    recommended = list(dict.fromkeys(combo_questions + topic_questions))[:8]

    theologian_info = [_get_author_meta(t) for t in req.theologians]
    return {
        "session_id": session.session_id,
        "theologians": theologian_info,
        "recommended_questions": recommended,
        "confession": session.confession,
        "confession_name": session.confession_name,
    }


@app.post("/api/symposium/ask")
async def symposium_ask(req: SymposiumAskRequest, request: Request):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    _check_session_owner(session, request)

    add_message(req.session_id, "user", req.message)

    # 신앙고백서 컨텍스트 검색 (세션에 confession이 있으면)
    confession_hits = []
    if session.confession:
        try:
            confession_hits = search(req.message, "confessions", CHROMA_DIR, top_k=5)
        except Exception:
            logger.warning("고백서 검색 실패 (session=%s…)", req.session_id[:8])
            confession_hits = []

    async def event_generator():
        for author_key in session.theologians:
            # 클라이언트 이탈 시 남은 신학자 호출 중단(구독 낭비·락 점유 방지, 2b)
            if await request.is_disconnected():
                logger.info("클라이언트 disconnect — 남은 신학자 호출 중단")
                break
            meta = _get_author_meta(author_key)
            try:
                hits = search(req.message, author_key, CHROMA_DIR, top_k=5)
            except Exception:
                logger.warning("검색 실패 (author=%s) — 근거 없이 진행", author_key)
                hits = []

            prompt = _build_theologian_prompt(meta, req.message, hits, session.history, confession_hits, session.confession_name)

            try:
                answer = await _call_claude(prompt)
            except Exception:
                # 예외 원문을 사용자 스트림에 싣지 않는다(내부 정보 유출 차단, 4b)
                logger.exception("답변 생성 실패 (author=%s)", author_key)
                answer = "(응답 생성에 실패했습니다.)"

            add_message(req.session_id, "theologian", answer, speaker=author_key, name_ko=meta["name_ko"])

            sources = [
                {
                    "title": h["metadata"].get("title", "?"),
                    "page": h["metadata"].get("page", "?"),
                    "text": h["text"][:300],
                }
                for h in hits[:3]
            ]

            yield {
                "event": "theologian",
                "data": json.dumps({
                    "speaker": author_key,
                    "name_ko": meta["name_ko"],
                    "tradition": meta.get("tradition", ""),
                    "text": answer,
                    "sources": sources,
                }, ensure_ascii=False),
            }

        yield {"event": "done", "data": json.dumps({"done": True})}

    return EventSourceResponse(event_generator())


@app.post("/api/symposium/direct")
async def symposium_direct(req: SymposiumDirectRequest, request: Request):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    _check_session_owner(session, request)
    if req.target not in session.theologians:
        raise HTTPException(400, f"{req.target}은 이 향연에 초대되지 않았습니다.")

    add_message(req.session_id, "user", req.message)

    # 신앙고백서 컨텍스트
    confession_hits = []
    if session.confession:
        try:
            confession_hits = search(req.message, "confessions", CHROMA_DIR, top_k=5)
        except Exception:
            logger.warning("고백서 검색 실패 (session=%s…)", req.session_id[:8])
            confession_hits = []

    meta = _get_author_meta(req.target)
    try:
        hits = search(req.message, req.target, CHROMA_DIR, top_k=5)
    except Exception:
        logger.warning("검색 실패 (author=%s) — 근거 없이 진행", req.target)
        hits = []

    prompt = _build_theologian_prompt(meta, req.message, hits, session.history, confession_hits, session.confession_name)
    answer = await _call_claude(prompt)

    add_message(req.session_id, "theologian", answer, speaker=req.target, name_ko=meta["name_ko"])

    sources = [
        {
            "title": h["metadata"].get("title", "?"),
            "page": h["metadata"].get("page", "?"),
            "text": h["text"][:300],
        }
        for h in hits[:3]
    ]

    return {
        "speaker": req.target,
        "name_ko": meta["name_ko"],
        "text": answer,
        "sources": sources,
    }
