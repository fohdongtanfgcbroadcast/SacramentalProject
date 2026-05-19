"""마스터플랜 비전 페이지 드리프트 점검.

static/masterplan.html 은 수작업 정적 HTML(의도된 비전 문서)이라 신학자
인제스트 시 자동 반영되지 않는다. 이 스크립트는 메타데이터 yaml(= /api/authors
가 읽는 권위 소스)과 masterplan.html 의 카드 목록을 비교해, 인제스트됐는데
비전 페이지에 카드가 없는 신학자(=드리프트, 수동 카드 추가 필요)를 보고한다.

사용: python scripts/check_masterplan_drift.py
종료코드: 드리프트 있으면 1, 없으면 0 (CI/pre-commit 훅 연동 가능)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / "data" / "metadata"
MASTERPLAN = ROOT / "static" / "masterplan.html"


def ingested_names() -> dict[str, str]:
    """metadata yaml → {name_ko: key} (confessions 제외)."""
    out: dict[str, str] = {}
    for yml in sorted(META_DIR.glob("*.yaml")):
        if yml.stem == "confessions":
            continue
        meta = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        a = meta.get("author", {})
        nk = a.get("name_ko", yml.stem)
        out[nk] = a.get("key", yml.stem)
    return out


def masterplan_card_names() -> set[str]:
    html = MASTERPLAN.read_text(encoding="utf-8")
    return {
        re.sub(r"\s+", " ", m.group(1)).strip()
        for m in re.finditer(r'<div class="card-name">([^<]+)</div>', html)
    }


def main() -> int:
    ing = ingested_names()
    cards = masterplan_card_names()

    missing = sorted(nk for nk in ing if nk not in cards)
    # 비전에만 있고 미인제스트(정상 — 의도된 미래 신학자): 참고용
    vision_only = sorted(c for c in cards if c not in ing)

    print(f"인제스트(metadata) {len(ing)}명 / masterplan 카드 {len(cards)}장")
    if missing:
        print(f"\n⚠ 드리프트: 인제스트됐으나 masterplan 카드 없음 ({len(missing)}명)")
        for nk in missing:
            print(f"  - {nk} ({ing[nk]})  → masterplan.html 해당 시대 섹션에 카드 추가 필요")
    else:
        print("\n✓ 드리프트 없음 — 인제스트된 모든 신학자가 masterplan에 반영됨")
    if vision_only:
        print(f"\n(참고) 비전 전용 — masterplan에만 있고 미인제스트 ({len(vision_only)}명, 정상):")
        print("  " + ", ".join(vision_only))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
