"""신앙고백서 영문 원본 → 한글 전문 번역 (.ko.txt 생성).

기존에 애드혹으로 만들던 .ko.txt 를 재현 가능하게 스크립트화한다.
긴 문서는 문단 경계로 청크 분할 후 claude CLI(--print, 구독 요금제)로
순차 번역하고, 부분 결과를 .ko.partial 에 증분 저장한다(중단 복구용).

사용:
    python scripts/translate_confessions.py <stem> [<stem> ...]
    python scripts/translate_confessions.py --all-new      # 신규 6개
예:
    python scripts/translate_confessions.py lambeth_articles arminian_articles
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from symposium.ingest import clean_text, extract_text_file  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "confessions"

# 신규 6개 (영문 → 파일 stem)
NEW_STEMS = [
    "lambeth_articles",
    "arminian_articles",
    "first_london_baptist",
    "irish_articles",
    "westminster_larger_catechism",
    "luthers_large_catechism",
]

# 청크 크기(자). claude --print 한 번에 안정적으로 번역되는 범위.
CHUNK_CHARS = 5500
CALL_TIMEOUT = 600  # 초

INSTRUCTION = """다음 영문 기독교 신앙고백서/교리문답 본문을 한국어로 충실하게 완역하라.

규칙:
- 본문 전체를 빠짐없이 한국어로 번역한다. 요약·생략·의역 금지.
- 장/조항/문답 번호와 제목, 구조를 원문 그대로 유지한다.
- 원문에 라틴어·헬라어·네덜란드어가 영어와 병기되어 있으면, 그 내용을 한국어로 한 번만 옮긴다(다른 언어 원문을 그대로 출력하지 말 것).
- 중요한 신학 용어는 한국어(원어) 형태로 병기할 수 있다. 예: 칭의(justification).
- 성경 인용·구절 표기는 그대로 옮긴다.
- 절대 메타 설명을 붙이지 말 것. "다음은 번역입니다", "이 텍스트는 ~의 일부입니다" 같은 문장 금지.
- 오직 번역된 본문만 출력한다. 머리말·꼬리말·해설 금지.

번역할 본문:
---
"""


# 정제 로직 단일 출처: src/symposium/textclean.py (측정·제거·인제스트 공유)
from symposium.textclean import (  # noqa: E402
    _IA_JUNK,
    _IA_TAIL,
    _JUNK_RE,
    _NAV_LABELS,
    _cut_at,
    _drop_boilerplate_lines,
    IA_STEMS,
    strip_ia_scan,
    strip_web_chrome,
)


def call_claude(prompt: str) -> str:
    """claude CLI --print 동기 호출. web.py 의 _call_claude 와 동일 규약."""
    proc = subprocess.run(
        ["claude", "--print"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("claude 빈 응답")
    return out


def chunk_by_paragraph(text: str, size: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            para = text.rfind("\n\n", start, end)
            if para > start + size // 2:
                end = para + 2
            else:
                nl = text.rfind("\n", start, end)
                if nl > start + size // 2:
                    end = nl + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end
    return chunks


def source_file(stem: str) -> Path:
    for ext in (".html", ".htm", ".txt"):
        p = RAW_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem} 원문 없음 (.html/.htm/.txt)")


def translate_stem(stem: str) -> None:
    src = source_file(stem)
    raw = extract_text_file(src)
    raw = strip_ia_scan(raw) if stem in IA_STEMS else strip_web_chrome(raw)
    text = clean_text(raw)
    chunks = chunk_by_paragraph(text)
    print(f"\n=== {stem} : {len(text):,}자 → {len(chunks)}청크 ({src.name}) ===")

    partial = RAW_DIR / f"{stem}.ko.partial"
    done_chunks: list[str] = []
    start_idx = 0
    if partial.exists():
        done_chunks = partial.read_text(encoding="utf-8").split("\n<<<CHUNK>>>\n")
        done_chunks = [c for c in done_chunks if c.strip()]
        start_idx = len(done_chunks)
        print(f"  부분 결과 발견: {start_idx}/{len(chunks)} 청크 이미 완료, 이어서 진행")

    # 일시적 사용량/속도 한도 대응: 점증 백오프로 throttle 창을 넘긴다.
    backoffs = [30, 90, 240, 480, 900]
    for i in range(start_idx, len(chunks)):
        ko = None
        for attempt in range(1, len(backoffs) + 2):
            try:
                ko = call_claude(INSTRUCTION + chunks[i] + "\n---")
                break
            except Exception as e:
                print(f"  [{i+1}/{len(chunks)}] 시도{attempt} 실패: {e}")
                if attempt > len(backoffs):
                    raise
                wait = backoffs[attempt - 1]
                print(f"    {wait}s 대기 후 재시도")
                time.sleep(wait)
        done_chunks.append(ko)
        partial.write_text("\n<<<CHUNK>>>\n".join(done_chunks), encoding="utf-8")
        print(f"  [{i+1}/{len(chunks)}] ✓ {len(ko):,}자")
        time.sleep(1)

    final = RAW_DIR / f"{stem}.ko.txt"
    final.write_text("\n\n".join(done_chunks).strip() + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"  완료 → {final.name} ({final.stat().st_size:,}B)")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    stems = NEW_STEMS if args == ["--all-new"] else args
    for stem in stems:
        translate_stem(stem)
    print("\n전체 완료.")


if __name__ == "__main__":
    main()
