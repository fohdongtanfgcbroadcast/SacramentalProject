"""EEBO-TCP TEI P5 XML → plain text 추출.

배경: EEBO-TCP (Early English Books Online — Text Creation Partnership)
는 1473~1700 영국 출판물을 사람이 직접 keyboarded 한 TEI P5 XML 코퍼스다.
Phase 1(2015 PD) + Phase 2(2020 PD) 모두 CC0 1.0 으로 공개돼 있고,
GitHub textcreationpartnership/<ID> 저장소에서 직접 받을 수 있다.

본 스크립트는 단일 TEI XML 을 받아 본문(`<text>` 하위)을 plain text 로
추출한다. djvu OCR(블랙레터 typeface 깨짐)을 대체할 깔끔한 영문 본문
공급용. vermigli 등 16세기 개혁파 영역본 코퍼스를 정상화하기 위한 도구.

규약:
- 본문(`<text>`) 만 추출, `<teiHeader>` 는 제외(서지정보)
- `<note>` (각주/방주) 는 본문 흐름을 끊지 않도록 별도 줄로 보존 ('[주: ...]'
  표기). 사용자가 인용 가치 있다고 판단했음(설교 본문 등엔 직접 인용 포함)
- `<gap>` (illegible) 는 '…'로 치환
- `<pb n="...">` (page break) 는 빈 줄로 치환 (인제스트가 어차피 가상
  3000자 페이지로 재분할)
- long-s ʃ/ſ → s 정규화 (검색·임베딩에 long-s 별 변종은 잡음)
- 그 외 Early Modern English 표기(v/u, j/i, &amp;, &c.)는 보존 —
  역사 자료 무결성 우선

사용:
  .venv/bin/python scripts/extract_eebo_tcp.py <input.xml> <output.txt>

산출: 출력 파일 + stdout 통계(문자수/줄수/자음 5+ 연속 단어 수 = OCR 깨짐
검출 시그니처).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from lxml import etree  # 빠르고 XPath 우수
    _USE_LXML = True
except ImportError:  # 표준 라이브러리 폴백
    import xml.etree.ElementTree as etree  # type: ignore
    _USE_LXML = False

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


def _qn(tag: str) -> str:
    """짧은 태그 이름을 { namespace }tag 로 확장."""
    return f"{{{TEI_NS}}}{tag}"


def extract_body(xml_path: Path) -> str:
    """TEI 파일에서 본문 텍스트를 추출한다.

    재귀 순회로 텍스트 노드를 수집하되 특수 요소(note/gap/pb)는 본문
    흐름을 손상하지 않게 별도 처리한다. teiHeader 는 통째로 skip.
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    text_el = root.find(_qn("text")) if not _USE_LXML else root.find("tei:text", NS)
    if text_el is None:
        raise SystemExit(f"{xml_path}: <text> 요소 없음")

    out: list[str] = []

    def walk(el) -> None:
        tag = etree.QName(el.tag).localname if _USE_LXML else el.tag.split("}")[-1]

        if tag == "note":
            # 각주/방주: 본문에 인라인 삽입하면 검색이 깨짐 → 별도 줄로
            note_text = "".join(el.itertext())
            note_text = re.sub(r"\s+", " ", note_text).strip()
            if note_text:
                out.append(f"\n[주: {note_text}]\n")
            if el.tail:
                out.append(el.tail)
            return

        if tag == "gap":
            out.append("…")
            if el.tail:
                out.append(el.tail)
            return

        if tag == "pb":  # page break
            out.append("\n\n")
            if el.tail:
                out.append(el.tail)
            return

        if tag in ("head", "p", "lg", "l", "div", "list", "item",
                   "table", "row", "cell", "ab", "speaker", "sp"):
            # 블록 요소는 줄바꿈으로 경계
            if el.text:
                out.append(el.text)
            for child in el:
                walk(child)
            out.append("\n")
            if el.tail:
                out.append(el.tail)
            return

        # 인라인 요소(hi, emph, foreign, name, ref, …) 또는 unknown:
        # 내용만 풀어서 흐름 유지
        if el.text:
            out.append(el.text)
        for child in el:
            walk(child)
        if el.tail:
            out.append(el.tail)

    walk(text_el)
    return "".join(out)


# Early Modern English 정규화
# - long-s(ſ/ʃ) → s : OCR 깨짐 아님, 정확한 transcription 의 활자 표기.
#   검색·임베딩에 별 변종으로 남으면 잡음 → s 로 합친다
# - 그 외 v/u, j/i, &amp;, &c. 는 역사 자료 보존 원칙으로 유지
_LONG_S_RE = re.compile(r"[ſʃ]")


def normalize(text: str) -> str:
    text = _LONG_S_RE.sub("s", text)
    # 공백 정리 (clean_text 에서도 하지만 통계 정확도를 위해 한 번 더)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def quality_check(text: str) -> dict:
    """OCR 깨짐 의심 시그니처를 정량화 — djvu vs TEI 품질 비교용."""
    consec = re.findall(r"\b[A-Za-z]*[bcdfghjklmnpqrstvwxyz]{5,}[A-Za-z]*\b", text)
    nonalpha = sum(1 for c in text if not c.isspace() and not (c.isascii() and c.isalpha()))
    alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    return {
        "chars": len(text),
        "lines": text.count("\n"),
        "consonant_clusters": len(consec),
        "alpha": alpha,
        "nonalpha": nonalpha,
        "nonalpha_ratio": nonalpha / (alpha + nonalpha) if (alpha + nonalpha) else 0.0,
        "samples": consec[:8],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src = Path(argv[1])
    dst = Path(argv[2])
    if not src.exists():
        raise SystemExit(f"입력 없음: {src}")

    raw = extract_body(src)
    norm = normalize(raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(norm, encoding="utf-8")

    q = quality_check(norm)
    print(f"입력: {src}")
    print(f"출력: {dst}")
    print(f"  문자수: {q['chars']:,}")
    print(f"  줄수:   {q['lines']:,}")
    print(f"  alpha:  {q['alpha']:,}  /  nonalpha: {q['nonalpha']:,}  "
          f"(비알파율 {q['nonalpha_ratio']:.2%})")
    print(f"  OCR 깨짐 시그니처(자음 5+ 연속 단어): {q['consonant_clusters']:,}")
    print(f"  예: {q['samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
