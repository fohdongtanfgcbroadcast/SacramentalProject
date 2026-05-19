"""RAG 검색·인용 코퍼스(ChromaDB) 잡음 정량 측정 — 데이터 품질 정비 1단계.

배경: src/symposium/ingest.py 는 clean_text 만 적용하고
scripts/translate_confessions.py 의 strip_web_chrome / strip_ia_scan /
_drop_boilerplate_lines 를 쓰지 않는다. 따라서 .ko.txt(한글 표시)는
정제됐으나 검색/인용 코퍼스에는 디지털화 잡음·웹 chrome·네비 라벨이
그대로 임베딩됐을 가능성이 있다. 이 스크립트는 그 오염도를 컬렉션별로
정량화한다. (측정 전용 — 어떤 데이터도 수정하지 않는다.)

정렬: 오염도(junk 청크 비율) 내림차순. 상위 3개 컬렉션은 실제 잡음
청크 표본 3개를 보고서에 그대로 첨부한다.

사용: .venv/bin/python scripts/measure_corpus_noise.py
산출: docs/data_quality_audit_<오늘>.md + stdout 요약표
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import chromadb

# step 2 에서 실제 제거할 잡음과 측정을 심볼 수준에서 일치시킨다.
# (재구현하면 측정치가 거짓이 된다 — translate_confessions 의 정의를 그대로 사용)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_confessions import (  # noqa: E402
    _IA_JUNK,
    _JUNK_RE,
    _NAV_LABELS,
    _drop_boilerplate_lines,
)

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
REPORT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / f"data_quality_audit_{date.today().isoformat()}.md"
)

PAGE = 5000  # 컬렉션당 페이지네이션 (메모리 상한)
SHRINK_THRESHOLD = 0.30  # _drop_boilerplate_lines 로 30% 이상 줄면 "심한 오염"
OCR_NONALPHA = 0.50  # 비알파 비율 50% 초과 + 20자 초과 → OCR 깨짐 의심 라인


def is_junk_line(s: str) -> bool:
    """translate_confessions 라인레벨 잡음 판정과 동일 규약.

    strip_web_chrome/strip_ia_scan 은 원본파일 슬라이싱(Gutenberg
    START/END, CCEL breadcrumb, IA tail 위치) 기반이라 청크에는
    적용 불가. 청크 수준으로 전이 가능한 것은 라인레벨 계층뿐:
    _IA_JUNK, _JUNK_RE, _NAV_LABELS.
    """
    if not s:
        return False
    if s.lower() in _NAV_LABELS:
        return True
    return bool(_IA_JUNK.search(s) or _JUNK_RE.search(s))


def is_ocr_broken_line(s: str) -> bool:
    """OCR 깨짐 휴리스틱 — strip 로직 범위 밖(별도 축). step 2 범위 아님.

    공백 제외 20자 초과 라인에서 [^A-Za-z] 비율이 50% 초과면 의심.
    (코퍼스 소스는 대부분 영문 IA/NPNF2/ANCL — .ko.txt 는 별도)
    """
    t = s.strip()
    if len(t) <= 20:
        return False
    nonspace = re.sub(r"\s", "", t)
    if not nonspace:
        return False
    nonalpha = len(re.sub(r"[A-Za-z]", "", nonspace))
    return nonalpha / len(nonspace) > OCR_NONALPHA


def classify_chunk(text: str) -> dict:
    lines = text.split("\n")
    has_junk = any(is_junk_line(ln.strip()) for ln in lines)
    has_ocr = any(is_ocr_broken_line(ln) for ln in lines)
    cleaned = _drop_boilerplate_lines(text)
    shrink = (len(text) - len(cleaned)) / len(text) if text else 0.0
    return {
        "has_junk": has_junk,
        "has_ocr": has_ocr,
        "shrink": shrink,
        "severe": shrink > SHRINK_THRESHOLD,
    }


def scan_collection(col) -> dict:
    total = junk = ocr = severe = 0
    shrink_sum = 0.0
    samples: list[str] = []  # junk 청크 표본 (상위 컬렉션 증빙용)
    offset = 0
    while True:
        batch = col.get(limit=PAGE, offset=offset, include=["documents"])
        docs = batch.get("documents") or []
        if not docs:
            break
        for d in docs:
            if d is None:
                continue
            total += 1
            r = classify_chunk(d)
            if r["has_junk"]:
                junk += 1
                if len(samples) < 3:
                    samples.append(d)
            if r["has_ocr"]:
                ocr += 1
            if r["severe"]:
                severe += 1
            shrink_sum += r["shrink"]
        offset += len(docs)
        if len(docs) < PAGE:
            break
    pct = lambda n: (100.0 * n / total) if total else 0.0
    return {
        "name": col.name,
        "total": total,
        "junk_pct": pct(junk),
        "severe_pct": pct(severe),
        "ocr_pct": pct(ocr),
        "mean_shrink": (100.0 * shrink_sum / total) if total else 0.0,
        "samples": samples,
    }


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    cols = client.list_collections()
    print(f"스캔 대상: {len(cols)} 컬렉션\n")

    rows: list[dict] = []
    for col in cols:
        r = scan_collection(col)
        rows.append(r)
        print(
            f"  {r['name']:<22} {r['total']:>7,}청크  "
            f"junk {r['junk_pct']:5.1f}%  severe {r['severe_pct']:5.1f}%  "
            f"ocr {r['ocr_pct']:5.1f}%  평균축소 {r['mean_shrink']:4.1f}%"
        )

    rows.sort(key=lambda x: x["junk_pct"], reverse=True)
    grand = sum(r["total"] for r in rows)

    lines: list[str] = []
    lines.append(f"# 데이터 품질 감사 — RAG 코퍼스 잡음 측정 ({date.today().isoformat()})\n")
    lines.append(
        "## 배경\n\n"
        "`src/symposium/ingest.py` 는 `clean_text` 만 적용하고 "
        "`scripts/translate_confessions.py` 의 라인레벨 잡음 제거"
        "(`_IA_JUNK`/`_JUNK_RE`/`_NAV_LABELS`/`_drop_boilerplate_lines`)를 "
        "쓰지 않는다. 본 보고서는 그 결과 검색/인용 코퍼스에 남은 "
        "디지털화 잡음·웹 chrome·네비 라벨의 오염도를 컬렉션별로 정량화한다. "
        "**측정 전용 — 데이터 미변경.**\n"
    )
    lines.append(
        "## 측정 방법\n\n"
        f"- 전수 스캔: {len(rows)} 컬렉션 / 총 {grand:,} 청크 "
        "(샘플링 아님)\n"
        "- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` "
        "포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). "
        "측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장\n"
        f"- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 "
        f"{int(SHRINK_THRESHOLD*100)}% 초과 줄어드는 청크 비율 (심한 오염)\n"
        "- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율\n"
        f"- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 "
        f"{int(OCR_NONALPHA*100)}% 초과 라인 포함 청크. strip 로직이 "
        "다루지 않음 — 보존처리/별도 작업 대상\n\n"
        "> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 "
        "있다. \"오염 청크 비율\"로는 정직한 수치다(어느 사본이 검색돼도 "
        "잡음 노출). 중복 제거는 하지 않는다.\n"
    )
    lines.append("## 컬렉션별 오염도 (junk % 내림차순)\n")
    lines.append("| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['total']:,} | {r['junk_pct']:.1f} | "
            f"{r['severe_pct']:.1f} | {r['ocr_pct']:.1f} | {r['mean_shrink']:.1f} |"
        )

    lines.append("\n## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)\n")
    for r in rows[:3]:
        lines.append(f"\n### {r['name']} (junk {r['junk_pct']:.1f}%)\n")
        if not r["samples"]:
            lines.append("_(표본 없음)_\n")
            continue
        for i, s in enumerate(r["samples"], 1):
            snippet = s[:600] + ("…" if len(s) > 600 else "")
            lines.append(f"**표본 {i}:**\n```\n{snippet}\n```\n")

    lines.append(
        "\n## 다음 단계 (결정 보류 — 측정만 보고)\n\n"
        "본 보고서는 raw 측정치만 제시한다. 어떤 컬렉션을 재인제스트할지"
        "(임계값·범위)는 step 2 에서 사용자와 결정한다.\n"
    )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n보고서 → {REPORT}")
    print(f"총 {grand:,} 청크 / {len(rows)} 컬렉션")
    top = rows[0]
    print(
        f"최다 오염: {top['name']} junk {top['junk_pct']:.1f}% "
        f"(severe {top['severe_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()
