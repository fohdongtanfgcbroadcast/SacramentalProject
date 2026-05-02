"""향연 세션 관리 — in-memory dict."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class SymposiumSession:
    session_id: str
    theologians: list[str]  # author key 리스트
    history: list[dict] = field(default_factory=list)
    # history 항목: {"role": "user"|"theologian", "speaker"?: str, "name_ko"?: str, "text": str}


_sessions: dict[str, SymposiumSession] = {}


def create_session(theologians: list[str]) -> SymposiumSession:
    sid = uuid.uuid4().hex[:12]
    session = SymposiumSession(session_id=sid, theologians=theologians)
    _sessions[sid] = session
    return session


def get_session(session_id: str) -> SymposiumSession | None:
    return _sessions.get(session_id)


def add_message(session_id: str, role: str, text: str, speaker: str = "", name_ko: str = "") -> None:
    session = _sessions.get(session_id)
    if session is None:
        return
    entry: dict = {"role": role, "text": text}
    if speaker:
        entry["speaker"] = speaker
    if name_ko:
        entry["name_ko"] = name_ko
    session.history.append(entry)
