"""향연 세션 관리 — in-memory dict + SQLite write-through(재시작 생존).

메모리 dict 를 working store(빠름)로, SQLite 를 durable mirror(재시작 복원)로 쓴다.
단일 uvicorn 워커라 프로세스 간 동시성은 없다. 모든 DB 연산은 try/except 로 감싸
실패해도 인메모리 동작을 막지 않는다(백엔드 degrade). 공개 인터페이스는 불변.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from dataclasses import dataclass, field

from symposium.config import SESSIONS_DB

# 유휴 만료(초): 마지막 활동 이후 이 시간이 지나면 세션 폐기
SESSION_TTL_SECONDS = 6 * 3600
# 동시 보관 최대 세션 수(초과 시 가장 오래 미사용 세션부터 축출)
MAX_SESSIONS = 500
# 세션당 히스토리 최대 메시지 수(프롬프트 크기 폭증·메모리 무한 누적 방지)
MAX_HISTORY = 40


@dataclass
class SymposiumSession:
    session_id: str
    theologians: list[str]  # author key 리스트
    confession: str = ""  # 토론 대상 신앙고백서 파일명 (비어있으면 일반 향연)
    confession_name: str = ""  # 한국어 제목
    owner: str = ""  # 세션 소유자 신원(JWT sub). 외부 게이트 경로에서만 채워짐(로컬=빈값)
    history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    # history 항목: {"role": "user"|"theologian", "speaker"?: str, "name_ko"?: str, "text": str}


_sessions: dict[str, SymposiumSession] = {}

# ─── SQLite 지속화(write-through) ───────────────────────────────
_DB_PATH = str(SESSIONS_DB)  # 테스트에서 monkeypatch 가능(임시 DB로 격리)
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "session_id TEXT PRIMARY KEY, theologians TEXT, confession TEXT, "
            "confession_name TEXT, owner TEXT, history TEXT, "
            "created_at REAL, last_active REAL)"
        )
        _conn.commit()
    return _conn


def _persist(s: SymposiumSession) -> None:
    try:
        _db().execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?)",
            (s.session_id, json.dumps(s.theologians), s.confession, s.confession_name,
             s.owner, json.dumps(s.history, ensure_ascii=False), s.created_at, s.last_active),
        )
        _db().commit()
    except Exception:
        pass


def _forget(session_id: str) -> None:
    try:
        _db().execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        _db().commit()
    except Exception:
        pass


def restore() -> int:
    """서버 시작 시 만료되지 않은 세션을 DB에서 인메모리로 복원. 복원 수 반환."""
    n = 0
    try:
        now = _now()
        cur = _db().execute(
            "SELECT session_id, theologians, confession, confession_name, owner, "
            "history, created_at, last_active FROM sessions"
        )
        for sid, theo, conf, confn, owner, hist, created, last in cur.fetchall():
            if now - (last or 0) > SESSION_TTL_SECONDS:
                _forget(sid)
                continue
            _sessions[sid] = SymposiumSession(
                session_id=sid,
                theologians=json.loads(theo) if theo else [],
                confession=conf or "",
                confession_name=confn or "",
                owner=owner or "",
                history=json.loads(hist) if hist else [],
                created_at=created or now,
                last_active=last or now,
            )
            n += 1
    except Exception:
        pass
    return n


def _now() -> float:
    return time.time()


def _purge_expired(now: float | None = None) -> None:
    """유휴 TTL 초과 세션을 정리(메모리+DB)."""
    now = _now() if now is None else now
    stale = [sid for sid, s in _sessions.items()
             if now - s.last_active > SESSION_TTL_SECONDS]
    for sid in stale:
        del _sessions[sid]
        _forget(sid)


def create_session(theologians: list[str], confession: str = "", confession_name: str = "", owner: str = "") -> SymposiumSession:
    _purge_expired()
    # 개수 상한: 초과 시 가장 오래 미사용(LRU) 세션부터 축출 → 무제한 생성으로 인한 메모리 고갈 방지
    while len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k].last_active)
        del _sessions[oldest]
        _forget(oldest)
    # 고엔트로피 세션 ID(~144비트). 기존 uuid4().hex[:12](48비트)는 열거 위험이 있었음.
    sid = secrets.token_urlsafe(18)
    session = SymposiumSession(session_id=sid, theologians=theologians, confession=confession, confession_name=confession_name, owner=owner)
    _sessions[sid] = session
    _persist(session)
    return session


def get_session(session_id: str) -> SymposiumSession | None:
    session = _sessions.get(session_id)
    if session is None:
        return None
    if _now() - session.last_active > SESSION_TTL_SECONDS:
        del _sessions[session_id]
        _forget(session_id)
        return None
    session.last_active = _now()
    return session


def add_message(session_id: str, role: str, text: str, speaker: str = "", name_ko: str = "") -> None:
    session = _sessions.get(session_id)
    if session is None:
        return
    session.last_active = _now()
    entry: dict = {"role": role, "text": text}
    if speaker:
        entry["speaker"] = speaker
    if name_ko:
        entry["name_ko"] = name_ko
    session.history.append(entry)
    # 히스토리 상한: 최근 MAX_HISTORY개만 유지
    if len(session.history) > MAX_HISTORY:
        del session.history[:-MAX_HISTORY]
    _persist(session)
