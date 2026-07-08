"""향연 세션 관리 — in-memory dict (TTL·개수·히스토리 상한 적용)."""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

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
    history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    # history 항목: {"role": "user"|"theologian", "speaker"?: str, "name_ko"?: str, "text": str}


_sessions: dict[str, SymposiumSession] = {}


def _now() -> float:
    return time.time()


def _purge_expired(now: float | None = None) -> None:
    """유휴 TTL 초과 세션을 정리."""
    now = _now() if now is None else now
    stale = [sid for sid, s in _sessions.items()
             if now - s.last_active > SESSION_TTL_SECONDS]
    for sid in stale:
        del _sessions[sid]


def create_session(theologians: list[str], confession: str = "", confession_name: str = "") -> SymposiumSession:
    _purge_expired()
    # 개수 상한: 초과 시 가장 오래 미사용(LRU) 세션부터 축출 → 무제한 생성으로 인한 메모리 고갈 방지
    while len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k].last_active)
        del _sessions[oldest]
    # 고엔트로피 세션 ID(~144비트). 기존 uuid4().hex[:12](48비트)는 열거 위험이 있었음.
    sid = secrets.token_urlsafe(18)
    session = SymposiumSession(session_id=sid, theologians=theologians, confession=confession, confession_name=confession_name)
    _sessions[sid] = session
    return session


def get_session(session_id: str) -> SymposiumSession | None:
    session = _sessions.get(session_id)
    if session is None:
        return None
    if _now() - session.last_active > SESSION_TTL_SECONDS:
        del _sessions[session_id]
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
