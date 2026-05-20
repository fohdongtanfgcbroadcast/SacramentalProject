# Changelog

이 프로젝트의 주요 변경 사항을 기록한다. (이전 변경은 git 로그 참조)

## [0.8.0] - 2026-05-20

### Added — wesley 정본 회복 (Step 5 ①, Class B 해소)

stub 책 3종(`sermons.txt`/`christian_perfection.txt`/`journal_vol1.txt`,
각 28KB CCEL Work Info HTML stub) 을 CCEL cache plain text 엔드포인트
의 정본으로 교체했다. 모두 Public Domain.

- `https://ccel.org/ccel/w/wesley/sermons/cache/sermons.txt` — 4.0MB
- `https://ccel.org/ccel/w/wesley/perfection/cache/perfection.txt` — 0.2MB
- `https://ccel.org/ccel/w/wesley/journal/cache/journal.txt` — 0.94MB
  (실질 일지 전체 합본, 파일명은 기존 키 유지 `journal_vol1.txt`)
- 기존 stub 은 `*.ccel_stub_legacy.html` 로 백업 (삭제 X).
- `data/metadata/wesley.yaml`: 3개 work 항목에 edition/source/license/note
  추가, 기존 `sermons_vol2`/`earnest_appeal` 은 유지.

### Changed — wesley 재인제스트로 본문 회복

- 1,921 → **12,407 청크** (+10,486, 6.5배).
- 측정 v5 (`docs/data_quality_audit_2026-05-20_v5.md`): junk % 0.0 유지,
  ocr % 0.9 → 8.7 증가는 CCEL 정본의 그리스어/라틴어 인용·각주·메타
  헤더에 따른 측정 휴리스틱 산물(검색 품질 무해).
- 잔여 상위 오염: confessions 3.4%, kierkegaard 3.3%, edwards 3.2%,
  luther 2.6% — 모두 Class A(재인제스트로 해소 가능).

## [0.7.0] - 2026-05-20

### Changed — bunyan 재인제스트 (Step 5 ④, 누락 보정)

v3 에서 bunyan(junk 4.0%) 이 새 상위 오염 1위가 됐는데, 진단 결과 코드
변경 불필요 — Step 2 strip(`textclean.strip_web_chrome`, 커밋 `da11213`)
의 Gutenberg `*** START`/`*** END` 마커 슬라이싱이 이미 잡아냄. bunyan
은 단지 Step 2 적용 *이전* 인제스트가 남아있었고, Step 3 에서 junk>5%
임계값으로 골랐을 때 4.0% 인 bunyan 이 제외돼 누락된 것.

- 재인제스트: 2,538 → 2,426 청크 (-112, strip 자연 감소).
- 측정 v4 (`docs/data_quality_audit_2026-05-20_v4.md`):
    bunyan junk % 4.0 → **0.0**, severe % 1.7 → 0.0.
- 새 상위 1위: confessions 3.4% (NewAdvent CCEL + Gutenberg 혼합,
  동일 패턴 — 동일 처리로 해소 가능).

## [0.6.0] - 2026-05-20

### Added — EEBO-TCP 도입 + vermigli OCR 손상 근본 해결 (Step 5 ①)

v2 의 새 상위 오염 1위 vermigli(junk 4.2%, OCR Class C — 블랙레터
djvu 깨짐, strip 무효) 를 EEBO-TCP 의 사람이 직접 keyboarded 한 TEI P5
transcription 으로 교체했다.

- `scripts/extract_eebo_tcp.py` 신규: EEBO-TCP A14350.xml(TEI P5,
  CC0 1.0) → plain text. long-s `ſ/ʃ` → `s` 정규화, `<note>` 별도 줄,
  `<gap>` → '…', Early Modern English 표기(v/u, j/i, &c.) 보존.
  향후 다른 EEBO-TCP 책에도 재사용 가능.
- `data/raw/vermigli/A14350.xml` (14.3MB), `common_places.txt`
  (10.3MB) 추가. 기존 djvu OCR 두 권은 `*.djvu_legacy.txt` 로 백업
  (삭제 X).
- `data/metadata/vermigli.yaml`: works 항목을 `common_places_vol1/2`
  djvu → 단일 `common_places` (EEBO-TCP A14350, CC0-1.0) 로 교체.

### Changed — vermigli 재인제스트

- ChromaDB vermigli 컬렉션 삭제 후 정제된 TEI 본문으로 재인제스트:
  21,507 청크 → 21,593 청크 (-1.5% 자연 변동).
- 측정 v3 (`docs/data_quality_audit_2026-05-20_v3.md`):
    junk % 4.2 → **0.0**, severe % 0.0 → 0.0,
    원본 단위 OCR 깨짐 시그니처(자음 5+ 연속 단어) **57,666 → 372 (-99.4%)**.
  ocr % 2.5 → 6.5 증가는 EEBO-TCP `<foreign>`/`<note>` 보존에 따른
  정상 라틴어 인용·각주 — 손상 아님.
- 새 상위 오염 1위: bunyan 4.0% (Gutenberg license boilerplate, Class A).

## [0.5.0] - 2026-05-20

### Added — 데이터 품질 감사 v2 (Step 4 재측정)

- `docs/data_quality_audit_2026-05-20.md`: Step 2(strip 단일 출처 통합,
  커밋 `da11213`) + Step 3(junk>5% 4개 컬렉션 재인제스트) 효과를 v1
  동일 스크립트·동일 정의로 재측정. 전수 스캔 599,435 청크 / 58 컬렉션.

### Changed — junk>5% 4개 컬렉션 임계값 아래로 진입

ingest 에 strip 적용 후 재인제스트한 결과 (청크 데이터만 갱신, 코드
변경 없음):

| 컬렉션 | v1 junk % | v2 junk % | Δ |
|---|--:|--:|--:|
| murray | 9.3 | 0.2 | -9.1pp |
| wesley | 7.5 | 0.0 | -7.5pp |
| zinzendorf | 6.4 | 0.0 | -6.4pp |
| watts | 5.2 | 0.0 | -5.2pp |

### Notes — 잔여 과제(이번 범위 밖, 별도 결정 대기)

- vermigli 4.2% 가 새 상위 1위 (OCR long-s 디지털화 잡음 — strip 무효
  Class C)
- wesley = Class B: HTML stub 으로 받힌 책 3종(`sermons.txt`,
  `christian_perfection.txt`, `journal_vol1.txt`)은 junk 는 없어졌지만
  실본문 결손 — Gutenberg 정본 재취득 필요
- zinzendorf · watts = Class C: 청크 잡음 0% 라도 OCR 손상(long-s ſ,
  단어 깨짐) 잔존 — 더 나은 스캔 소스 필요/수용 결정

## [0.4.0] - 2026-05-19

### Added — 교부 6명 추가 (51 → 57, "교부 시대" 대폭 확대)

공개도메인 영역본(ANCL / NPNF2 / Oxford Library of Fathers)에서:
- 유스티누스 (100–165) — 호교론·트리포와의 대화, ANCL, 1,977청크
- 알렉산드리아의 클레멘스 (150–215) — 권면·교사·양탄자, ANCL, 4,511청크
- 키프리아누스 (200–258) — 서신·교회일치론, ANCL, 4,606청크
- 예루살렘의 키릴로스 (313–386) — 교리강해, Oxford LF, 1,730청크
- 암브로시우스 (340–397) — 직무·성령·신앙론, NPNF2-10, 5,182청크
- 예로니무스 (342–420) — 서간·불가타·논쟁서, NPNF2-06, 5,234청크

metadata yaml 6종 신규. 보류(소스 접근 한계): 나지안조스의 그레고리오스
(NPNF2-07 Cyril 합본), 니사의 그레고리오스(NPNF2-05 IA djvu 반복 실패).
중세 스콜라(아벨라르·위클리프·후스·롬바르두스·오컴): 공개도메인 영역본
부재로 미진행(라틴 원전/저작권 현대역만).

### Added — masterplan 드리프트 점검 스크립트

- `scripts/check_masterplan_drift.py`: metadata yaml(= /api/authors 권위
  소스) vs static/masterplan.html 카드 비교. 인제스트됐으나 비전 페이지
  카드 없는 신학자를 보고, 드리프트 시 exit 1 (CI/훅 연동 가능).
  방금 발생한 "신규 신학자 masterplan 미반영" 버그의 재발 방지 장치.

### Fixed

- `static/masterplan.html`: 교부 시대 섹션에 신규 10명(이번+직전 배치)
  카드 생몰순 반영 — 드리프트 0 확인(check_masterplan_drift exit 0).

## [0.3.0] - 2026-05-19

### Added — 교부 4명 추가 (47 → 51, 목표 "교부 시대" 보강)

공개도메인 영역본(Ante-Nicene Christian Library / NPNF2)에서 핵심 교부 추가:
- 이레네우스 (130–202) — 이단 반박, ANCL Writings of Irenaeus 1·2, 4,745청크
- 테르툴리아누스 (155–220) — ANCL Writings of Tertullian 1·2·3, 7,207청크
- 오리게네스 (184–253) — ANCL Writings of Origen 1·2 (원리론·켈수스 반박), 5,211청크
- 대 바실리오스 (330–379) — NPNF2-08 Letters and Select Works, 8,048청크

metadata yaml 4종 신규(`irenaeus/tertullian/origen/basil.yaml`).
- 나지안조스의 그레고리오스: 단독 PD 영역본 부재 + NPNF2-07(Cyril 합본)
  정제본도 평탄화로 신뢰성 있는 분리 불가 → 보류(소스 접근 한계).

### Added — 신학자 추가: 울리히 츠빙글리 (46 → 47)

종교개혁 3대 신학자(루터·칼뱅·츠빙글리) 중 누락되어 있던 츠빙글리를 추가.
공개도메인 영역본(Internet Archive, Jackson 편): Selected Works(1901),
Latin Works and Correspondence Vol.1·2(1912). `zwingli` 컬렉션 4,116청크.
`data/metadata/zwingli.yaml` 신규.

- 부처(1491–1551)·슈페너(1635–1705): 공개도메인 영역본 부재로 미추가
  (저작권 현대역/구매 필요) — 48번째 신학자 슬롯은 사용자 자료 의존.

### Added — 기존 신앙고백서 한글 갭 해소

원래 21개 중 .ko.txt 없던 2개를 한글 전문 번역:
- 하이델베르크 교리문답 (1563) — 한국 개혁교회 핵심 문서
- 제1 스위스 신앙고백 (1536)
→ 신앙고백서 30개 중 28개 한글 전문 (영문만: 제네바·케임브리지 OCR 2개).

### Added — 신앙고백서 확장 (21 → 30, 마스터플랜 목표 달성)

CCEL Schaff·Project Gutenberg·OPC·The Reformed Reader·Internet Archive
공개 도메인 소스에서 9개 신앙고백서를 추가했다.

한글 전문 번역 완료 (7개):
- 루터 대교리문답 (1529, Project Gutenberg)
- 성공회 교리문답 (1549/1662, CCEL Schaff) — 공동기도서 수록
- 램버스 신조 (1595, CCEL Schaff) — 39개조의 칼뱅주의적 부록
- 아르미니우스 항론파 5개조 (1610, CCEL Schaff) — 도르트 신조가 반박한 문서
- 아일랜드 신조 (1615, CCEL Schaff) — 어셔, 웨스트민스터의 주요 원천
- 제1 런던 침례교 신앙고백 (1644, The Reformed Reader)
- 웨스트민스터 대요리문답 (1647, OPC)

제네바 교리문답(1545, IA 1815 영역본)·케임브리지 강령(1648, IA 1850판)도
IA 스캔 chrome 제거 핸들러 추가 후 한글 전문 번역 완료.
→ **신앙고백서 30/30 전부 한글 전문**(웹 /api 검증: 한글 30, 영문 0).
  단 두 건은 IA 스캔 특성상 시대 편집 서문/광고 일부 포함.

`confessions` 컬렉션: 4,695 → 6,470 청크 (신규 +1,775, 9개 문서).

### Added — 번역 파이프라인 스크립트화

- `scripts/translate_confessions.py` 신규: 영문 신앙고백서 → 한글 전문 번역.
  - 청크 분할 후 Claude CLI(`--print`, 구독 요금제) 순차 번역
  - 소스별 웹 chrome 제거: Project Gutenberg(START/END), CCEL Schaff(브레드크럼/푸터),
    OPC, The Reformed Reader + 범용 boilerplate 라인 필터
  - `.ko.partial` 증분 저장으로 중단/실패 시 재개
  - 점증 백오프 재시도(30→90→240→480→900s)로 사용량 한도 대응
- 기존 애드혹 한글 번역 방식을 재현 가능한 스크립트로 대체

### Fixed

- `static/masterplan.html`: 신규 교부 4명(이레네우스·테르툴리아누스·오리게네스·
  대 바실리오스) 카드 누락 수정 — 비전 페이지가 수동 유지보수 정적 HTML이라
  교부 추가 시 미반영되던 문제. 교부 시대 섹션에 생몰순으로 추가(101→105 카드).
  ※ 근본 구조: 비전 문서는 의도적으로 정적(미인제스트 신학자 포함)이라
    /api 연동 대신 기존 패턴(수동 카드 추가) 유지. 향후 추가 시 동일 갱신 필요.

### Changed

- `scripts/download_confessions.py`: 9개 항목 추가, 미확보 소스 각주 갱신
- `data/confessions.md`: 로드맵 v3 — RAG ✅/한글 🇰🇷 표시, 보류 사유 명기

### Known Issues / Notes

- 제네바·케임브리지: IA 스캔본 한글 전문 번역 완료(IA chrome 핸들러 `strip_ia_scan`
  추가). 단 1815/1850 편집 서문·출판 광고 일부가 번역에 포함됨(IA 스캔 특성,
  핵심 본문은 정상). 추후 정밀 소스 확보 시 교체 여지.
- Claude 구독 사용량 윈도우 한도: 대량 연속 번역 시 ~30분+ 차단 발생 가능
  (백오프로 일부 대응, 대규모 작업은 분할/시간 분산 권장).
- 제외(저작권): 제2차 바티칸 공의회 문헌, 벨하르 신앙고백 / 저작권 모호: 바르멘 선언
- 협화신조서(1580)는 구성요소가 이미 개별 수록되어 중복 — 미추가
