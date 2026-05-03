"""신앙고백서/신조 텍스트 자동 다운로드.

CCEL (Schaff's Creeds of Christendom), Project Gutenberg, 기타 공개 도메인 소스에서
신앙고백서 텍스트를 다운로드하고 메타데이터 YAML을 생성한다.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import yaml

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# --- 다운로드 대상 정의 ---
# 카테고리별 정리. 각 항목: (filename, title, title_ko, year, url, tradition)

CONFESSIONS = [
    # ─── 고대 공의회 신조 (1~5세기) ───
    ("didache.html", "Didache (Teaching of the Twelve Apostles)",
     "디다케", 100,
     "https://www.newadvent.org/fathers/0714.htm",
     "초대교회"),

    ("ecumenical_creeds.html", "Ecumenical Creeds (Apostles', Nicene, Athanasian, Chalcedonian)",
     "공의회 신조 (사도·니케아·아타나시우스·칼케돈)", 325,
     "https://ccel.org/ccel/schaff/creeds2/creeds2.iv.i.html",
     "초대교회 공의회"),

    # ─── 종교개혁 — 루터교 ───
    ("luthers_small_catechism.txt", "Luther's Small Catechism",
     "루터 소교리문답", 1529,
     "https://www.gutenberg.org/cache/epub/1670/pg1670.txt",
     "루터교"),

    ("augsburg_confession.txt", "Augsburg Confession",
     "아우크스부르크 신앙고백", 1530,
     "https://www.gutenberg.org/cache/epub/275/pg275.txt",
     "루터교"),

    ("apology_augsburg.txt", "Apology of the Augsburg Confession",
     "아우크스부르크 신앙고백 변증론", 1531,
     "https://www.gutenberg.org/cache/epub/6744/pg6744.txt",
     "루터교"),

    ("smalcald_articles.txt", "Smalcald Articles",
     "슈말칼덴 조항", 1537,
     "https://www.gutenberg.org/cache/epub/273/pg273.txt",
     "루터교"),

    ("formula_of_concord.html", "Formula of Concord",
     "일치신조", 1577,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iii.iv.html",
     "루터교"),

    # ─── 종교개혁 — 개혁파 ───
    ("first_helvetic_confession.html", "First Helvetic Confession",
     "제1 스위스 신앙고백", 1536,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.iv.html",
     "개혁파"),

    ("gallican_confession.html", "Gallican Confession (French Confession of Faith)",
     "갈리아 신앙고백", 1559,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.vii.html",
     "개혁파"),

    ("scots_confession.html", "Scots Confession",
     "스코틀랜드 신앙고백", 1560,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.ix.html",
     "개혁파"),

    ("belgic_confession.html", "Belgic Confession",
     "벨직 신앙고백", 1561,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.viii.html",
     "개혁파"),

    ("heidelberg_catechism.html", "Heidelberg Catechism",
     "하이델베르크 교리문답", 1563,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.vi.html",
     "개혁파"),

    ("second_helvetic_confession.html", "Second Helvetic Confession",
     "제2 스위스 신앙고백", 1566,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.v.html",
     "개혁파"),

    # ─── 성공회 ───
    ("thirty_nine_articles.html", "Thirty-Nine Articles of Religion",
     "39개 신앙조항", 1571,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.xi.html",
     "성공회"),

    # ─── 개혁파 정통 (17세기) ───
    ("canons_of_dort.html", "Canons of Dort",
     "도르트 신조", 1619,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.xvi.html",
     "개혁파"),

    ("westminster_confession.html", "Westminster Confession of Faith",
     "웨스트민스터 신앙고백", 1646,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.xvii.html",
     "장로교"),

    ("westminster_shorter_catechism.html", "Westminster Shorter Catechism",
     "웨스트민스터 소요리 문답", 1647,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.iv.xviii.html",
     "장로교"),

    ("savoy_declaration.html", "Savoy Declaration",
     "사보이 선언", 1658,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.v.i.i.html",
     "회중교회"),

    ("second_london_baptist.html", "Second London Baptist Confession",
     "제2 런던 침례교 신앙고백", 1689,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.v.ii.i.html",
     "침례교"),

    # ─── 감리교 ───
    ("methodist_articles.html", "Methodist Articles of Religion",
     "감리교 25개 신앙조항", 1784,
     "https://ccel.org/ccel/schaff/creeds3/creeds3.v.vi.html",
     "감리교"),

    # ─── 현대 ───
    ("lausanne_covenant.html", "Lausanne Covenant",
     "로잔 언약", 1974,
     "https://lausanne.org/content/covenant/lausanne-covenant",
     "복음주의"),
]


def download_file(url: str, dest: Path) -> bool:
    """URL에서 텍스트/HTML 파일 다운로드. 이미 있으면 스킵."""
    if dest.exists() and dest.stat().st_size > 100:
        print(f"    [있음] {dest.name}")
        return True

    print(f"    다운로드: {url}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Symposium/0.2 (theology RAG research)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            dest.write_bytes(content)
        print(f"    ✓ {len(content):,} bytes → {dest.name}")
        time.sleep(1.5)  # 예의상 대기
        return True
    except Exception as e:
        print(f"    ✗ 실패: {e}")
        return False


def generate_yaml(successful_works: list[dict], metadata_dir: Path) -> None:
    """신앙고백서 메타데이터 YAML 생성."""
    yaml_path = metadata_dir / "confessions.yaml"
    data = {
        "author": {
            "key": "confessions",
            "name_ko": "신앙고백서·신조",
            "name_en": "Creeds and Confessions",
            "born": 100,
            "tradition": "교회 공의회·교파",
        },
        "works": successful_works,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  YAML 생성: {yaml_path.name} ({len(successful_works)}개 고백서)")


def main():
    raw_dir = DATA_ROOT / "raw" / "confessions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = DATA_ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0
    successful_works = []

    print(f"\n{'='*60}")
    print(f"  신앙고백서·신조 다운로드")
    print(f"  대상: {len(CONFESSIONS)}개")
    print(f"{'='*60}")

    for filename, title_en, title_ko, year, url, tradition in CONFESSIONS:
        dest = raw_dir / filename
        if download_file(url, dest):
            successful_works.append({
                "file": filename,
                "title": f"{title_ko} — {title_en}",
                "year": year,
                "tradition": tradition,
            })
            total_ok += 1
        else:
            total_fail += 1

    if successful_works:
        generate_yaml(successful_works, metadata_dir)

    print(f"\n{'='*60}")
    print(f"  완료: {total_ok}개 다운로드, {total_fail}개 실패")
    print(f"{'='*60}")

    # 미포함 안내
    print("\n[참고] 소스 미확보로 미포함:")
    print("  - 제네바 교리문답 (1542) — CCEL/Gutenberg에 없음")
    print("  - 웨스트민스터 대요리 문답 (1647) — Schaff에 별도 항목 없음")
    print("  - 바르멘 선언 (1934) — 공개 URL 미발견")
    print("  - 벨하르 신앙고백 (1982) — 저작권")
    print("  - 제2차 바티칸 공의회 문헌 — 저작권")


if __name__ == "__main__":
    main()
