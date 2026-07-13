"""measure_corpus_noise 측정 휴리스틱 회귀 테스트.

- is_ocr_broken_line: 한글(CJK) 본문 오탐 방지(moltmann 99.5% 오탐 건),
  기존 영문 판정 의미는 불변.
- report_path: --suffix 인자로 같은 날 재실행 시 덮어쓰기 방지.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from measure_corpus_noise import is_ocr_broken_line, report_path  # noqa: E402


class TestIsOcrBrokenLine:
    def test_normal_english_not_flagged(self):
        assert not is_ocr_broken_line(
            "The grace of our Lord Jesus Christ be with you all."
        )

    def test_garbled_symbols_flagged(self):
        # 비알파 기호/숫자 과다 라인 — 기존 판정 유지
        assert is_ocr_broken_line("~~*** 34,, ..;; ^^!! 78 ##(( ))__ 12 ::")

    def test_short_line_not_flagged(self):
        assert not is_ocr_broken_line("*** 34 ;; ^^")

    def test_korean_text_not_flagged(self):
        # moltmann 컬렉션 오탐 건: 한글은 알파벳으로 취급해야 한다
        assert not is_ocr_broken_line(
            "하나님의 은혜와 우리 주 예수 그리스도의 평강이 너희 모든 사람과 함께 있을지어다."
        )

    def test_korean_with_punctuation_not_flagged(self):
        assert not is_ocr_broken_line(
            "희망의 신학은 종말론을 신학의 중심으로 되돌려 놓았다(몰트만, 1964)."
        )

    def test_hanja_mixed_korean_not_flagged(self):
        assert not is_ocr_broken_line("십자가에 달리신 하나님(神)의 삼위일체론적 이해와 고난")

    def test_korean_garbled_symbols_still_flagged(self):
        # 한글이 섞여도 기호가 지배적이면 여전히 깨짐 판정
        assert is_ocr_broken_line("은혜 ~~*** 34,, ..;; ^^!! 78 ##(( ))__ 12 ::")


class TestReportPath:
    def test_default_path_unchanged(self):
        p = report_path()
        assert p.name.startswith("data_quality_audit_")
        assert p.name.endswith(".md")
        assert p.parent.name == "docs"

    def test_suffix_appended_before_extension(self):
        assert report_path("v2").name == report_path().name.replace(
            ".md", "_v2.md"
        )
