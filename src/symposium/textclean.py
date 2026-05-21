"""소스 chrome/boilerplate 제거 — 측정·제거·인제스트 공유 단일 정의.

extract_text_file 직후·clean_text 이전의 raw 텍스트에 적용한다
(Gutenberg START/END 마커 등 슬라이싱이 원본 라인에 의존하므로 청크 후엔 불가).

이 모듈이 단일 출처(single source of truth)다:
- src/symposium/ingest.py     — RAG 코퍼스 정제
- scripts/translate_confessions.py — .ko.txt 한글 표시 정제
- scripts/measure_corpus_noise.py  — 잡음 측정(제거 대상과 정의 일치 보장)
"""
from __future__ import annotations

import re

_NAV_LABELS = {
    "home", "documents", "confessions", "library", "biography", "history",
    "beliefs", "share", "this page using:", "tweet", "contents", "loading…",
    "loading...", "what we believe", "about the opc", "our standards",
    "general assembly", "worldwide outreach", "news", "churches", "standards",
    "new horizons", "ordained servant", "site map", "contact us",
    "committed to historic baptist & reformed", "the reformed reader home page",
    "gospel tracts", "christian education", "foreign missions", "home missions",
    "ministries", "historian", "short-term missions", "disaster response",
}
_JUNK_RE = re.compile(
    r"@.*\.(edu|com|org|net)\b"
    r"|\b(Surface Mail|Phone|Fax)\s*:"
    r"|Project Gutenberg"
    r"|All Rights Reserved"
    r"|^\s*©"
    r"|Please login"
    r"|register to save highlights"
    r"|VIEWNAME|workSection"
    r"|Christian Classics Ethereal Library"
    r"|The Reformed Reader"
    r"|file:///ccel/"
    r"|https?://",
    re.I,
)


def _cut_at(text: str, pattern: str) -> str:
    """pattern 에 처음 매칭되는 줄에서 본문을 잘라낸다(그 줄 이후 전부 제거)."""
    rx = re.compile(pattern, re.I)
    out: list[str] = []
    for ln in text.split("\n"):
        if rx.search(ln.strip()):
            break
        out.append(ln)
    return "\n".join(out).strip()


def _drop_boilerplate_lines(text: str) -> str:
    """소스별 슬라이스 후 남은 잡라인 제거 + 앞뒤 잡라인 트림."""
    kept: list[str] = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if s.lower() in _NAV_LABELS or _JUNK_RE.search(s):
            continue
        kept.append(ln)
    # 앞뒤 빈 줄 트림
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


_UNDERSCORE_RE = re.compile(r"^\s*_{30,}\s*$")
_CCEL_REF_URL_RE = re.compile(r"^\s*\d+\.\s+file:///ccel/")
# CCEL 자동생성 색인 블록 헤더 (본문 뒤 푸터)
_CCEL_FOOTER_HEAD_RE = re.compile(
    r"^\s*(Indexes|Index of Scripture References|Index of Citations"
    r"|Index of Pages|Index of Names|Index of Subjects"
    r"|Scripture Index|Greek Words and Phrases|Hebrew Words and Phrases)\s*$",
    re.I,
)
# CCEL 푸터 직전의 책 자체 back matter(서지·일반 색인). 후반부에서만 푸터로 인정.
# bonaventure/minds_road_to_god.txt 는 'SELECTED BIBLIOGRAPHY'가 482/1941(25%)
# 의 프론트매터(역자 서문)라 전역 매칭 시 본문이 통째로 파괴됨 → 위치 게이트 필수.
_CCEL_BACKMATTER_RE = re.compile(r"^\s*((SELECTED\s+)?BIBLIOGRAPHY|Index of .+)\s*$", re.I)


def _strip_ccel_cache(lines: list[str]) -> str:
    """CCEL cache plain text 헤더/푸터 제거.

    구조:
      ___________________   ← 첫 underscore 구분선
      Title: …
      Creator(s): …
      CCEL Subjects: …      ← 메타블록
      ...
      ___________________   ← 두 번째 underscore 구분선 (본문 시작 직전)
      본문 …
      ___________________   ← 마지막 underscore (선택)
      This document is from the Christian Classics Ethereal Library …
      References
       1. file:///ccel/…    ← 푸터 URL 리스트

    헤더: 처음 두 underscore 사이를 메타블록으로 보고 그 뒤부터 본문.
    푸터: 'This document is from the Christian Classics' 또는 References+file:///ccel/
    URL 블록 시작점에서 절단.
    """
    n = len(lines)
    underscore_idx = [i for i, l in enumerate(lines) if _UNDERSCORE_RE.match(l)]

    # 헤더 슬라이싱: 첫 두 underscore 가 책의 앞 1/3 안에 있을 때만 메타블록으로 해석
    # (책 안의 구분선 underscore 와 구분)
    body_start = 0
    if len(underscore_idx) >= 2 and underscore_idx[1] < max(50, n // 3):
        body_start = underscore_idx[1] + 1
    elif len(underscore_idx) >= 1 and underscore_idx[0] < max(20, n // 10):
        body_start = underscore_idx[0] + 1

    body_end = n

    # 푸터 시작점: CCEL 색인 블록('Indexes'/'Index of …') · ThML notice ·
    # References+file:///ccel/ URL 리스트 중 본문에서 가장 먼저 나오는 마커.
    # 그 위 underscore 구분선까지 통째로 절단(색인 헤더 위 구분선 포함).
    for i in range(body_start, n):
        low_i = lines[i].lower()
        if (_CCEL_FOOTER_HEAD_RE.match(lines[i])
                or "this document is from the christian classics" in low_i
                or _CCEL_REF_URL_RE.match(lines[i])
                or (i > n // 2 and _CCEL_BACKMATTER_RE.match(lines[i]))):
            cut = i
            for j in range(i, body_start, -1):
                if _UNDERSCORE_RE.match(lines[j]):
                    cut = j
                    break
            body_end = cut
            break

    return "\n".join(lines[body_start:body_end]).strip()


def strip_web_chrome(raw: str) -> str:
    """소스 사이트 boilerplate(네비게이션·푸터)를 제거해 본문만 남긴다.

    extract_text_file 직후, clean_text 이전의 원문에 적용한다
    (Gutenberg START/END 마커가 살아 있어야 슬라이스 가능).
    """
    lines = raw.split("\n")
    low = raw.lower()

    # 1) Project Gutenberg: START/END 마커 사이만
    if "project gutenberg" in low:
        s = next((i for i, l in enumerate(lines)
                  if "START OF" in l.upper() and "GUTENBERG" in l.upper()), None)
        e = next((i for i, l in enumerate(lines)
                  if "END OF" in l.upper() and "GUTENBERG" in l.upper()), None)
        if s is not None and e is not None and e > s:
            body = "\n".join(lines[s + 1:e]).strip()
            body = _cut_at(body, r"This text was converted|Project Wittenberg|Walther Library")
            return _drop_boilerplate_lines(body)

    # 1b) CCEL cache plain text: '___...___' 구분선 + 'CCEL Subjects:' 메타 블록
    # (Schaff HTML 출처와 구분 — cache 는 plain text 라 'Prev/Next' 브레드크럼 없음)
    if "ccel subjects:" in low or (
        "this document is from the christian classics ethereal library" in low
    ):
        return _drop_boilerplate_lines(_strip_ccel_cache(lines))

    # 2) CCEL Schaff: 앞뒤 "« Prev … Next »" 브레드크럼 / 푸터 사이
    if "christian classics ethereal library" in low or "creeds of christendom" in low:
        bc = [i for i, l in enumerate(lines) if "Prev" in l and "Next" in l]
        start = bc[0] + 1 if bc else 0
        end = len(lines)
        for i in range(start, len(lines)):
            s = lines[i].strip()
            if (s.startswith("Please login") or s.startswith("VIEWNAME")
                    or s == "workSection"
                    or ("Prev" in s and "Next" in s)):
                end = i
                break
        return _drop_boilerplate_lines("\n".join(lines[start:end]).strip())

    # 3) OPC: 본문 "Q. 1." 부터 푸터(© / 대문자 메뉴) 직전까지
    if "orthodox presbyterian church" in low:
        start = next((i for i, l in enumerate(lines)
                      if re.match(r"^Q\.?\s*1\.", l.strip())), 0)
        body = "\n".join(lines[start:]).strip()
        body = _cut_at(
            body,
            r"^The Orthodox Presbyterian Church$|^\+?\d[\d ]{6,}$"
            r"|Contact Form|^Find a Church$|^ABOUT US$|^© 20",
        )
        return _drop_boilerplate_lines(body)

    # 4) The Reformed Reader: 고백서 제목부터 푸터 직전까지
    if "the reformed reader" in low:
        start = next((i for i, l in enumerate(lines)
                      if "CONFESSION OF FAITH" in l.upper() or "A.D. 1644" in l), 0)
        end = len(lines)
        for i in range(start, len(lines)):
            s = lines[i].strip()
            if (s.startswith("Copyright 1999")
                    or s.startswith("The Reformed Reader Home")
                    or s in ("Share", "This Page Using:")):
                end = i
                break
        return _drop_boilerplate_lines("\n".join(lines[start:end]).strip())

    return _drop_boilerplate_lines(raw)


# Internet Archive djvu.txt 스캔본 스템 (콘텐츠 마커가 OCR로 깨져 스템 기반 처리)
IA_STEMS = {"geneva_catechism", "cambridge_platform"}

# OCR에 관대한 느슨한 부분문자열 패턴(디지털화/도서관/보존 trailer)
_IA_JUNK = re.compile(
    r"internet\s*arc|arc[bh]ive\.org|\bN\W?R\W?L\W?F\b|microsoft\s*corp"
    r"|librar[yi]\s*of\s*congress|united\s*states\s*of\s*america"
    r"|berkeley\s*librar|deacidified|bookkeeper|neutralizing\s*agent"
    r"|treatment\s*date|preservation\s*tech|paper\s*preservation"
    r"|cranberry\s*township|funding\s*from",
    re.I,
)
_IA_TAIL = re.compile(
    r"deacidified|preservation\s*tech|neutralizing\s*agent|treatment\s*date"
    r"|paper\s*preservation|berkeley\s*librar|librar[yi]\s*of\s*congress"
    r"|irregular\s*verbs|german\s*derivatives|principal\s*terminations",
    re.I,
)


def strip_ia_scan(raw: str) -> str:
    """IA djvu.txt: 디지털화/도서관 스탬프(head) + 보존 trailer(tail) 제거.

    OCR 노이즈에 강하도록 느슨한 부분문자열 패턴 사용. 잡라인 제거 후
    앞쪽의 자투리(짧은 표제 조각)는 첫 실질 산문까지 트림한다.
    """
    lines = raw.split("\n")
    # tail: 보존/도서관 trailer 가장 앞 지점에서 절단(앞 1/3 이후만 탐색해 본문 보호)
    cut = len(lines)
    for i in range(len(lines) // 3, len(lines)):
        if _IA_TAIL.search(lines[i]):
            cut = i
            break
    kept = [l for l in lines[:cut] if not _IA_JUNK.search(l)]
    # head: 첫 '실질 산문'(글자 25자 이상)까지 자투리 제거
    start = 0
    for i, l in enumerate(kept):
        if len(re.sub(r"[^A-Za-z]", "", l)) >= 25:
            start = i
            break
    return _drop_boilerplate_lines("\n".join(kept[start:]).strip())
