"""FastAPI 웹 서버 — 신학 문헌 RAG 플랫폼."""
from __future__ import annotations

import asyncio
import json
import subprocess
import unicodedata
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from symposium.config import CHROMA_DIR, METADATA_DIR
from symposium.retrieve import search
from symposium.session import create_session, get_session, add_message

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="Symposium", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# claude CLI 동시 실행 방지 — 순차 처리 큐
_claude_lock = asyncio.Lock()

SYSTEM_INSTRUCTION = (
    "당신은 기독교 조직신학 전문 연구 조수입니다. "
    "아래 참고 자료에 근거해 질문에 답하세요. "
    "각 주장 뒤에 출처를 [저작명, p.페이지] 형식으로 표기하고, "
    "중요한 신학 용어는 원어(독일어/영어/라틴어)를 병기하세요. "
    "자료에 직접 언급되지 않은 내용은 '자료에는 직접 언급되지 않음'이라고 명시하세요. "
    "마크다운 서식(#, **, *, - 등)을 사용하지 말고 일반 텍스트로만 답변하세요."
)

THEOLOGIAN_SYSTEM_TEMPLATE = (
    "당신은 {name_ko}입니다. {tradition} 전통의 신학자로서, "
    "당신의 저작에 근거하여 답변하세요. "
    "이전 대화 맥락을 참고하되, 다른 신학자의 견해에 동의하거나 반박할 수 있습니다. "
    "각 주장 뒤에 출처를 [저작명, p.페이지] 형식으로 표기하세요. "
    "중요한 신학 용어는 원어(독일어/영어/라틴어/그리스어)를 병기하세요. "
    "자료에 직접 언급되지 않은 내용은 '자료에는 직접 언급되지 않음'이라고 명시하세요. "
    "마크다운 서식(#, **, *, - 등)을 사용하지 말고 일반 텍스트로만 답변하세요."
)


# --- Models ---

class SearchRequest(BaseModel):
    query: str
    author: str = "moltmann"
    top_k: int = 5


class AskRequest(BaseModel):
    question: str
    author: str = "moltmann"
    top_k: int = 5


class AuthorInfo(BaseModel):
    key: str
    name_ko: str
    work_count: int


class SymposiumStartRequest(BaseModel):
    theologians: list[str]
    topic: str = ""

class SymposiumAskRequest(BaseModel):
    session_id: str
    message: str

class SymposiumDirectRequest(BaseModel):
    session_id: str
    message: str
    target: str


# --- Helpers ---

def _get_available_authors() -> list[AuthorInfo]:
    """메타데이터 YAML이 있는 저자 목록 반환."""
    import yaml

    authors = []
    for yml in sorted(METADATA_DIR.glob("*.yaml")):
        with open(yml, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        author_info = meta.get("author", {})
        works = meta.get("works", [])
        authors.append(AuthorInfo(
            key=author_info.get("key", yml.stem),
            name_ko=author_info.get("name_ko", yml.stem),
            work_count=len(works),
        ))
    return authors


def _build_context(question: str, hits: list[dict]) -> str:
    """검색 결과를 claude CLI에 전달할 프롬프트로 조합."""
    parts = [SYSTEM_INSTRUCTION, "", "# 참고 자료", ""]
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        source = f"[{m.get('title', '?')}, p.{m.get('page', '?')}]"
        parts.append(f"### 발췌 {i} {source}\n{hit['text']}\n")
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
) -> str:
    """특정 신학자의 관점으로 답변할 프롬프트 구성."""
    system = THEOLOGIAN_SYSTEM_TEMPLATE.format(
        name_ko=author_meta["name_ko"],
        tradition=author_meta.get("tradition", ""),
    )
    parts = [system, "", "# 참고 자료 (본인 저작에서 발췌)", ""]
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        source = f"[{m.get('title', '?')}, p.{m.get('page', '?')}]"
        parts.append(f"### 발췌 {i} {source}\n{hit['text']}\n")

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
    async with _claude_lock:
        proc = await asyncio.create_subprocess_exec(
            "claude", "--print",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=120,
        )
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")
            raise HTTPException(status_code=502, detail=f"Claude CLI 오류: {err[:500]}")
        return stdout.decode("utf-8").strip()


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


@app.post("/api/search")
async def api_search(req: SearchRequest):
    try:
        hits = search(req.query, req.author, CHROMA_DIR, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
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
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
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


@app.post("/api/symposium/start")
async def symposium_start(req: SymposiumStartRequest):
    if len(req.theologians) > 5:
        raise HTTPException(400, "최대 5명까지 선택 가능합니다.")
    if len(req.theologians) < 1:
        raise HTTPException(400, "최소 1명을 선택해야 합니다.")

    session = create_session(req.theologians)
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
    }


@app.post("/api/symposium/ask")
async def symposium_ask(req: SymposiumAskRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    add_message(req.session_id, "user", req.message)

    async def event_generator():
        for author_key in session.theologians:
            meta = _get_author_meta(author_key)
            try:
                hits = search(req.message, author_key, CHROMA_DIR, top_k=5)
            except Exception:
                hits = []

            prompt = _build_theologian_prompt(meta, req.message, hits, session.history)

            try:
                answer = await _call_claude(prompt)
            except Exception as e:
                answer = f"(응답 생성 실패: {e})"

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
async def symposium_direct(req: SymposiumDirectRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    if req.target not in session.theologians:
        raise HTTPException(400, f"{req.target}은 이 향연에 초대되지 않았습니다.")

    add_message(req.session_id, "user", req.message)

    meta = _get_author_meta(req.target)
    try:
        hits = search(req.message, req.target, CHROMA_DIR, top_k=5)
    except Exception:
        hits = []

    prompt = _build_theologian_prompt(meta, req.message, hits, session.history)
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
