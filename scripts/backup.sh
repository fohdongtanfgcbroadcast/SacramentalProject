#!/bin/bash
# Symposium 자산 백업 → 타임스탬프 tar.gz. launchd 주간 실행 또는 수동 호출.
#
# 기본(경량): data/metadata(컬렉션 정의) + Theology_export_word.json(용어집) +
#   data/recommended_questions.json + data/sessions.db(세션 상태). 수백 KB — 빠르고 저렴.
#   chroma_db 는 data/raw 에서 재인덱싱으로 재생성 가능한 파생 데이터라 기본에서 제외.
# 전체(SYMPOSIUM_BACKUP_FULL=1): 위 + data/raw(원천 텍스트 2.7G) + chroma_db(인덱스 9.3G).
#
# 백업 위치: $SYMPOSIUM_BACKUP_DIR (기본 ~/Backups/symposium). 최근 8개만 보관.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${SYMPOSIUM_BACKUP_DIR:-$HOME/Backups/symposium}"
mkdir -p "$DEST"
TS="$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"
CORE=(data/metadata Theology_export_word.json data/recommended_questions.json data/sessions.db)
FULL_EXTRA=(data/raw chroma_db)

TARGETS=()
for p in "${CORE[@]}"; do [ -e "$p" ] && TARGETS+=("$p"); done
if [ "${SYMPOSIUM_BACKUP_FULL:-0}" = "1" ]; then
  for p in "${FULL_EXTRA[@]}"; do [ -e "$p" ] && TARGETS+=("$p"); done
  OUT="$DEST/symposium_full_$TS.tar.gz"
else
  OUT="$DEST/symposium_core_$TS.tar.gz"
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "backup: 대상 없음 — 건너뜀" >&2
  exit 0
fi

tar -czf "$OUT" "${TARGETS[@]}"

# 보존 정책: core/full 각각 최근 8개만 유지
PREFIX="$(basename "$OUT" | sed 's/_[0-9]*_[0-9]*\.tar\.gz$//')"
ls -1t "$DEST/${PREFIX}"_*.tar.gz 2>/dev/null | tail -n +9 | while read -r old; do
  rm -f "$old"
done

echo "backup: $OUT ($(du -h "$OUT" | cut -f1))"
