# Changelog

이 프로젝트의 주요 변경 사항을 기록한다. (이전 변경은 git 로그 참조)

## [0.15.1] - 2026-07-22

### Changed — 정적 UI 색상 하드코딩 변수화 (다크모드 하드코딩 정리)

- `static/style.css`: `:root` 밖에 직접 박혀 있던 색상 11곳을 CSS 변수로 치환 — 신규 변수 7종
  (`--hero-top`, `--glow-1/2`, `--surface-raised`, `--border-strong`, `--text-on-accent`,
  `--gold-tint`). 대상: 히어로 그라디언트, body 글로우, confession-card 배경/보더,
  confession-context/truncated 골드 틴트, chat-msg-user·btn-primary 텍스트색,
  picker-btn/topic-card/theologian-card-link hover 배경.
- `static/masterplan.html`: 자체 `<style>` 블록의 하드코딩 4곳 변수화 — 신규 변수 4종
  (`--hero-top`, `--card-done`, `--glow-1/2`).
- 색상값 자체는 동일(시각 변화 0) — 향후 라이트/다크 테마 분기 시 `:root` 오버라이드만으로
  대응 가능한 기반. 검증: 두 파일 모두 `:root` 밖 하드코딩 색상 0건, 라이브 서빙 확인.
- `pyproject.toml` version 0.14.1로 뒤처져 있던 것 0.15.1로 동기화.

## [0.15.0] - 2026-07-22

### Added — 외부 도메인 접근 (symposium.nt-apparatus.com, SSO 인증 게이트)

- 사용자 결정으로 "로컬 전용"(0.13.0) 공식 해제 — 단 무인증 노출이 아니라 **Alexandria 계정
  SSO 게이트 뒤** 노출. cloudflared named tunnel ingress에 `symposium.nt-apparatus.com →
  localhost:8000` 추가 + `tunnel route dns` CNAME(config 백업 `~/.cloudflared/config.yml.bak_20260722`).
- **`_external_auth_gate` 미들웨어**: Host가 symposium.nt-apparatus.com일 때만 Alexandria JWT
  쿠키(access_token, HS256 동일 시크릿, plist env 주입) 검증 — stdlib hmac 검증(서명·exp·
  role∈approved/admin), 미인증 HTML→nt-apparatus.com/login 302·API→401, 시크릿 미설정 시
  fail-closed 503. **로컬(127.0.0.1) 직접 접속은 무게이트**(종전 사용성 보존).
  한계(명시): DB 미조회라 탈퇴자 즉시 차단은 안 됨 — 토큰 만료 7일 의존(소규모 수용).
- Alexandria 측: 로그인 쿠키 조건부 `domain=.nt-apparatus.com`(v9.95) + 서가 탭 링크
  `https://symposium.nt-apparatus.com/symposium` 상시 표시(로컬 접속 시 127.0.0.1로 자동 전환).
- 검증: 로컬 200 / 외부 비인증 401·302(login 리다이렉트) / 외부 인증 200 / 본체 도메인 무영향.

## [0.14.2] - 2026-07-22

### Changed — 원탁대화 Claude 호출 모델 핀 + 타임아웃 (Alexandria 연동)

- `_call_claude`: `--model claude-sonnet-4-6` 명시 핀. 기본모델(Opus) 상속 시 웜 상태에서도
  60초 상한 초과("응답 생성 시간 초과")로 원탁대화가 무동작이었음(2회 실측) + Opus 구독 소진.
  핀 후 실측 38.6초 정상 답변(칼뱅 페르소나·출처 3건).
- `_CLAUDE_TIMEOUT` 60→90초 (콜드스타트 마진).
- 운영: 무인증 퀵터널 `com.symposium.tunnel` 언로드+plist `.disabled`(2026-07-07 로컬전용 결정
  복원 — 재부팅 RunAtLoad로 되살아나 있었음). 외부 접근은 Alexandria(nt-apparatus.com)의
  인증 게이트 프록시 `/api/sym-rt/*` 경유로 일원화(Alexandria v9.92).

## [0.14.1] - 2026-07-14

### Fixed — 데이터 품질 측정 도구 (`scripts/measure_corpus_noise.py`)

- **한글(CJK) ocr% 오탐 수정**: OCR 깨짐 휴리스틱이 `[A-Za-z]` 외 전부를
  비알파로 세어 한글 본문이 통째로 깨짐 판정되던 문제. 알파벳 취급 범위에
  한글 음절/자모 + CJK 한자를 추가. moltmann(한글 번역 PDF) ocr
  99.5% → **2.6%**(진짜 잡음만 잔존), vermigli 6.5% → 5.9%(`[주: …]`
  각주 라벨의 한글 '주' 오탐 교정). **나머지 56개 컬렉션 측정치 불변**
  (v1/v2 보고서 diff 로 검증).
- **`--suffix` 인자 추가**: 같은 날 재실행 시 보고서
  (`docs/data_quality_audit_<날짜>.md`) 덮어쓰기 방지. 예:
  `--suffix v2` → `..._v2.md`.
- 회귀 테스트 9개 추가 (`tests/test_measure_corpus_noise.py`) — 총 24개.
- 데이터 품질 재측정 보고서 2종: 2026-07-14(수정 전) / _v2(수정 후).
  junk 트랙 완료 확정 — 최대 confessions 0.3%, 5/22 대비 드리프트 없음.

## [0.14.0] - 2026-07-08

### Security/Hardening — 자체 하드닝 (보안 감사 Phase 0C Track A, 사용자 승인)

로컬 전용 유지(외부 재노출 안 함) 결정에 따른 Symposium 자체 남용 방어선. 전 항목
TDD(회귀 테스트 15개 추가, `tests/`).

- **입력 제약**: `top_k` → `Field(ge=1, le=20)`, query/message 등 문자열 `max_length`.
  거대 top_k로 인한 대량 Chroma 쿼리·프롬프트 증폭 차단.
- **요청 본문 크기 제한**: 미들웨어에서 Content-Length 256KB 초과 시 413.
- **보안 응답 헤더**: 전 응답에 CSP(`default-src 'self'`…) + `X-Content-Type-Options`
  + `X-Frame-Options: DENY` + `Referrer-Policy` (XSS/클릭재킹 심층방어).
- **정적 XSS 교정**: `app.js`에 `escapeHtml` 추가·적용, `symposium.js` 신학자
  목록/추천질문 본문의 미이스케이프 innerHTML 싱크 교정.
- **세션 하드닝**(`session.py`): 유휴 TTL(6h) + 최대 세션 수(500, LRU 축출) +
  히스토리 상한(40) + 고엔트로피 세션 ID(`secrets.token_urlsafe`, 기존 48비트→~144비트).
  메모리 무한 증가·세션 열거 위험 완화.
- **전역 락 DoS 완화**(`_call_claude`): 락 대기자 상한(8) 초과 시 429, claude 타임아웃
  120s→60s, 타임아웃 시 `proc.kill()`로 좀비 프로세스(구독 요금 지속 소모) 회수.
- **경로 이탈 방어**: `/api/confession-text/{filename}`에 `_confession_path` 격리
  검증(resolve 후 부모 디렉터리 일치 강제). 프레임워크 라우팅 의존 제거.
- **예외 원문 유출 차단**: `/api/search`·`/api/ask`의 `detail=str(e)` 및 claude
  stderr 반환 제거 → 일반 메시지 + 서버 로깅.
- **관리 표면 축소**: `/docs`·`/redoc`·`/openapi.json` 비활성.
- **레거시 제거**: 미사용 `generate.py`(anthropic API 경로) 삭제 + `.env.example`의
  `ANTHROPIC_API_KEY` 제거. 미사용 import(`subprocess`·`unicodedata`) 정리.
- **테스트 인프라**: pyproject `[project.optional-dependencies] test` +
  `[tool.pytest.ini_options]`, `tests/`(conftest + 15 테스트) 신설.

### 남은 후속(별도 승인 필요)
- Phase 0C Track B(Alexandria 게이트 뒤 재노출)는 "로컬 전용 유지" 결정으로 보류.
- 데이터 연동(canon.db 교차참조) P1~P3은 감사 보고서 로드맵 참조.

## [0.13.0] - 2026-07-07

### Security — 긴급 격리 (보안 감사 Phase 0A/0B, 사용자 승인)

`docs/security/SECURITY_AUDIT_AND_INTEGRATION_2026-07-07.md` 감사에서 확인된
활성 Critical 3건(무인증 공개 노출 · 구독요금 도난/DoS · 프롬프트 인젝션→RCE)에
대한 즉시 격리.

- **공개 노출 차단**: `com.symposium.tunnel`(cloudflared quick tunnel) 언로드로
  `*.trycloudflare.com` 무인증 공개 URL 제거.
- **loopback 바인딩**: `serve.py`와 `com.symposium.server.plist`의
  `0.0.0.0` → `127.0.0.1`. LAN/Tailscale 직접 노출 차단(파일시스템 페더레이션은
  무영향). 향후 공개는 인증 게이트(Alexandria require_approved 프록시) 경유만.
- **RCE 근절**: `_call_claude`의 `claude --print` 호출에 `--tools ""` 추가.
  전역 `~/.claude/settings.json`(defaultMode=bypassPermissions + allow=Bash/Edit/…)을
  상속해 프롬프트 인젝션이 호스트 셸 실행으로 이어지던 경로를 구조적으로 차단.
  신학 답변은 순수 텍스트 생성이라 도구 불필요(실증 검증 완료).

### 남은 후속(별도 승인 필요)
- 인증 게이트 편입(Phase 0C), 레이트리밋·세션 TTL·top_k 상한, XSS/CSP 등은
  감사 보고서 Phase 0C 이후 로드맵 참조.

## [0.12.0] - 2026-05-22

### Fixed — 재인제스트 고아 청크 + 빈 컬렉션 사고 방지 (ingest.py)

**근본 원인**: `ingest_author` 가 `get_or_create_collection` + `upsert` 만
사용. 청크 ID 는 `{author}:{file}:{chunk_idx}:p{가상페이지}` 인데, strip 으로
텍스트가 짧아지면 가상 페이지 수가 줄어 **옛 인제스트의 뒷페이지 청크가 ID
충돌 없이 고아로 잔존**. 그 뒷페이지가 CCEL footer References(`file:///ccel/…`)
구간이라 junk 로 측정됨. (anselm 측정 2,758 vs 재인제스트 로그 2,392 — 차이
366개가 고아. junk 청크 page 185~197 > 새 페이지수 182 로 실증.)

- **삭제 후 재생성**: 재인제스트 시 기존 컬렉션을 `delete_collection` 후
  `create_collection`. 고아 청크 원천 차단.
- **빈 컬렉션 가드**: 유효한(실재) 소스 파일이 0개면 `SystemExit` 으로 중단,
  기존 컬렉션 보존. (moltmann raw 가 SecramentalProject 로의 깨진 심볼릭
  링크 → 0 청크로 컬렉션이 비는 사고 방지. 가드로 재발 차단.)

### Changed — junk>0 14개 재인제스트로 고아 junk 정리

anselm 8.6→0.0, eckhart 6.9→0.0, john_damascus 6.7→0.0, julian_norwich
3.8→0.0, wesley 2.8→0.0, luther 1.4→0.0, knox 1.4→0.0 (전부 0.0%).
moltmann 은 깨진 링크로 재인제스트 불가 → `data/processed/moltmann/*.jsonl`
(이전 청크 12,114개)에서 재임베딩으로 복원.

- 현재 로컬 코퍼스: **58 컬렉션 / 567,448 청크**, junk 0.0% 51개.
  잔여 junk ≤0.3%(confessions 0.3%, murray·cyril_jerusalem·moltmann 0.2%,
  kuyper·schweitzer·harnack 0.1%) — 다른 패턴 소량, 후속 과제.

## [0.11.0] - 2026-05-22

### Fixed — CCEL cache back matter strip 보완 (aquinas/nature_and_grace)

`textclean._strip_ccel_cache` 의 CCEL 푸터 컷이 CCEL 자동 `Indexes`
블록만 절단하고, 그 **앞**에 위치한 책 자체 back matter(서지 목록·일반
색인)는 본문에 남기던 문제를 보완.

- `_CCEL_BACKMATTER_RE` 추가: 전체 라인 `(SELECTED )?BIBLIOGRAPHY` ·
  `Index of …`. **위치 게이트 `i > n//2`** 로 후반부에서만 푸터로 인정.
  (bonaventure/minds_road_to_god 는 `SELECTED BIBLIOGRAPHY` 가
  482/1941≈25% 의 프론트매터[역자 서문]라 전역 매칭 시 본문 파괴 → 게이트 필수.)
- 7개 CCEL 소스 회귀 어서션 PASS: 6개 strip 출력 byte 동일,
  aquinas/nature_and_grace 만 -27,845자(`BIBLIOGRAPHY`·`Index of
  References to Other Authors and Sources` 색인 제거).

### Changed — bernard · aquinas · bonaventure 재인제스트

CCEL strip(분기 1b) + 위 보완 반영을 위해 3개 author 재인제스트.
현재 로컬 코퍼스 측정(`docs/data_quality_audit_2026-05-22.md`) 기준 junk:

- bernard 2,254청크 → **0.0%**
- aquinas 39,283청크 → **0.0%**
- bonaventure 5,152청크 → **0.0%**

검증 방식: claude·codex·gemini 3개 CLI 병렬 교차검증(읽기전용)으로
strip 로직 확인 후 보완·재인제스트.

## [0.10.0] - 2026-05-21

### Changed — Class A 2차 11개 일괄 재인제스트 (Step 5 ③ 2nd round)

v6 잔여 junk ≥ 0.5% 11개 컬렉션(whitefield · anselm · francis ·
john_damascus · basil · baxter · knox · rutherford · ambrose · eckhart ·
julian_norwich) 을 일괄 재인제스트. 코드 변경 없이 Step 2 누락 보정.

- 11개 모두 **정확히 0.0% 로 해소**:
    whitefield 1.7→0.0, anselm 1.4→0.0, francis 1.3→0.0,
    john_damascus 1.0→0.0, basil 1.0→0.0, baxter 0.9→0.0,
    knox 0.9→0.0, rutherford 0.9→0.0, ambrose 0.7→0.0,
    eckhart 0.5→0.0, julian_norwich 0.5→0.0
- 총 청크 607,137 → 606,511 (-626, strip 자연 감소).

### Notes — 새 발견: bernard 1.6% (CCEL header/footer 패턴)

v7 새 상위 1위 bernard 1.6%. 표본 분석 결과 strip 이 잡지 못하는
**CCEL 헤더(`Title:`/`Creator(s):`/`LC Call no:` 등 메타 블록) + 푸터
(`References` + `file:///ccel/.../cache/...html` URL 리스트)** 가 청크에
실려 있다. `_JUNK_RE` 의 `Christian Classics Ethereal Library` 가 일부
청크는 잡지만 메타 라인 다수는 매치 안 됨 — **strip 코드 보강 후보**.
aquinas 0.4% 도 동일 패턴(Gutenberg 헤더 메타 + ELECTRONIC EDITION
노트).

측정 v7: `docs/data_quality_audit_2026-05-21_v7.md`.

## [0.9.0] - 2026-05-21

### Changed — Class A 7개 일괄 재인제스트 (Step 5 ③, Step 2 누락 보정)

v5 잔여 상위 오염은 모두 bunyan 과 동일 패턴(Step 2 strip 적용 이전
인제스트가 남은 상태). 코드 변경 없이 일괄 재인제스트만으로 해소.

| 컬렉션 | 청크 변화 | junk % 변화 |
|---|---|---|
| confessions | 6,470 → 4,776 (-26%) | 3.4 → 0.3 |
| kierkegaard | 1,040 → 1,006 | 3.3 → 0.0 |
| edwards | 4,032 → 3,919 | 3.2 → 0.0 |
| luther | 14,941 → 14,433 | 2.6 → 0.0 |
| harnack | 5,841 → 5,530 | 2.1 → 0.1 |
| melanchthon | 1,541 → 1,503 | 2.1 → 0.0 |
| schweitzer | 3,253 → 3,193 | 2.1 → 0.1 |

총 청크 609,895 → 607,137 (-2,758, strip 자연 감소). confessions 가
-26% 로 가장 큰 감소(NewAdvent CCEL boilerplate 다수 제거).

### Notes — 코퍼스 전체 상태

이 시점부터 **junk % > 2% 컬렉션이 없다.** 새 상위 1위 whitefield 1.7%
(동일 Class A 패턴, 추가 일괄 재인제스트로 해소 가능). 남은 *구조적*
잡음은 OCR Class C(strip 무효): zinzendorf·watts(OCR 손상),
moltmann ocr 99.5%(한글 본문 오탐) 정도.

측정 v6: `docs/data_quality_audit_2026-05-21.md`.

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
