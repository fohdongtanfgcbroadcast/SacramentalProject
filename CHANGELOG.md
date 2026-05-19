# Changelog

이 프로젝트의 주요 변경 사항을 기록한다. (이전 변경은 git 로그 참조)

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
