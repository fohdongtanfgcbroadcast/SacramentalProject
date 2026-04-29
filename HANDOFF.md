# 다음 에이전트(맥의 Claude Code)를 위한 핸드오프

이 문서는 **사용자 PC(macOS)의 Claude Code 세션**에서 이 프로젝트를 이어받아 진행하기 위한 인수인계서입니다. 클라우드 샌드박스에서는 PDF 자료에 접근할 수 없으므로, 실제 인제스트와 검증은 **로컬 맥 환경**에서 진행합니다.

---

## 1. 프로젝트 한 줄 요약

위르겐 몰트만(Jürgen Moltmann) 전집(OCR PDF)을 ChromaDB에 임베딩해 두고, Claude Opus 4.7로 신학 질의응답을 하는 RAG 시스템.

상세 사양·디렉터리 구조·사용 예시는 `README.md` 참조.

---

## 2. 현재 상태 (2026-04-29 기준)

- ✅ 프로젝트 스캐폴딩 완료, `claude/theology-project-assessment-FqpYz` 브랜치에 푸시됨
- ✅ Python 패키지 `theology_rag` 구현: `config / ingest / retrieve / generate / cli`
- ✅ 메타데이터 템플릿 `data/metadata/moltmann.yaml` (몰트만 주요 8권)
- ⏳ **아직 안 된 것**: 실제 PDF 인제스트, 임베딩 생성, 질의 테스트, OCR 품질에 맞춘 정제 규칙 튜닝

---

## 3. 환경 준비 (맥에서 처음 1회)

### 3-1. 저장소 클론

```bash
git clone <repository-url> SecramentalProject
cd SecramentalProject
git checkout claude/theology-project-assessment-FqpYz
```

### 3-2. Python 환경 (uv 권장)

```bash
# uv가 없으면: brew install uv  또는  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e .
```

또는 pip:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 3-3. API 키

```bash
cp .env.example .env
# .env 를 열어서 ANTHROPIC_API_KEY=sk-ant-... 입력
```

> `theology-rag ask` 는 환경변수 `ANTHROPIC_API_KEY` 를 읽습니다. `.env` 자동 로딩이 필요하면 `python-dotenv` 추가 후 `cli.py` 상단에 `from dotenv import load_dotenv; load_dotenv()` 한 줄.

### 3-4. macOS 의존성 메모

- PyMuPDF (`pymupdf`): 휠 제공, Apple Silicon에서 문제 없음
- sentence-transformers + PyTorch: Apple Silicon은 자동으로 MPS 사용. 첫 실행 시 `BAAI/bge-m3` 모델(≈2GB) Hugging Face에서 다운로드.
- ChromaDB: 의존성에 SQLite 필요 (macOS 기본 포함)

---

## 4. PDF 배치

```
data/
└── raw/
    └── moltmann/
        ├── 01_희망의_신학.pdf
        ├── 02_십자가에_달리신_하나님.pdf
        └── ...
```

- `data/raw/` 는 `.gitignore` 에 포함 — 절대 커밋되지 않음
- `data/metadata/moltmann.yaml` 의 `file:` 필드를 **실제 PDF 파일명과 정확히 일치**시킬 것
- 파일명에 한글·공백·특수문자 다 OK (Python `pathlib` 가 처리)

---

## 5. 인제스트 실행

```bash
theology-rag ingest moltmann
```

흐름:
1. `data/metadata/moltmann.yaml` 로드 → 저작 목록
2. 각 PDF에 대해 PyMuPDF로 페이지별 텍스트 추출
3. `clean_ocr_text()` 로 정제 (하이픈 줄바꿈 병합, 페이지 번호 제거 등)
4. `chunk_text()` 로 청크 분할 (기본 800자, 200 오버랩, 문단·문장 경계 우선)
5. `BAAI/bge-m3` 로 임베딩 생성
6. `chroma_db/moltmann` 컬렉션에 업서트
7. 디버깅용 정제본을 `data/processed/moltmann/<파일명>.jsonl` 로 저장

진행상황은 stdout에 표시. 실패 시 어느 책에서 막혔는지 즉시 확인 가능.

---

## 6. 테스트

### 6-1. 검색만 (Claude 호출 없음, 빠르고 무료)

```bash
theology-rag search "십자가 신학" --author moltmann --top-k 5
```

- 결과 표가 출력됨: 저작·페이지·거리(distance)·발췌 미리보기
- **이걸로 임베딩 품질을 먼저 검증**해야 합니다. 거리 0.3~0.5 안쪽이면 양호, 그보다 멀면 청크가 너무 길거나 짧을 가능성

### 6-2. 검색 + Claude 답변

```bash
theology-rag ask "몰트만의 종말론에서 '희망'은 어떻게 정의되는가?"
```

- top-k 발췌를 컨텍스트로 Claude Opus 4.7 호출 (시스템 프롬프트는 캐싱됨)
- 토큰 사용량과 캐시 히트가 마지막 줄에 표시됨

---

## 7. 코드 구조 빠른 참조

| 파일 | 역할 | 손볼 가능성 높은 곳 |
|---|---|---|
| `src/theology_rag/config.py` | 상수 (모델·청크 크기) | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL` |
| `src/theology_rag/ingest.py` | PDF→청크→임베딩→Chroma | `clean_ocr_text()` (정제 규칙), `chunk_text()` (분할 로직) |
| `src/theology_rag/retrieve.py` | 쿼리→top-k | 필터 (저작별·연도별), 재랭킹 |
| `src/theology_rag/generate.py` | Claude 호출 | `SYSTEM_PROMPT`, 컨텍스트 포맷 |
| `src/theology_rag/cli.py` | typer CLI | 새 명령 추가 |
| `data/metadata/moltmann.yaml` | 저작 목록 | 보유 자료에 맞춰 수정 |

---

## 8. 튜닝 포인트 (실제 데이터 본 후)

### 8-1. OCR 정제

`clean_ocr_text()` 의 현재 규칙은 보수적입니다. 실제 OCR 품질을 본 뒤 다음을 추가/수정 검토:

- **머리글·바닥글 패턴 제거**: 모든 페이지에 반복되는 책 제목·장 제목·페이지 번호 패턴
- **한글 OCR 오류 빈도 높은 글자 보정** (예: "ㅇ"이 "0"으로 인식되는 경우)
- **각주 번호 정리**: 본문에 섞여 들어간 작은 숫자
- **단어 중간 공백 제거**: OCR이 "신 학" 처럼 자모 단위로 분리하는 경우

### 8-2. 청크 크기

기본 800자 / 200 오버랩은 일반적인 학술 산문에 적합. 몰트만 본문이:
- 긴 논증 단위 → 1200~1500자로 키우기
- 짧은 문단·격언 → 500~600자로 줄이기

### 8-3. 검색 품질이 나쁘면

먼저 `theology-rag search` 로 발췌 품질 확인:
1. 의미는 맞는데 동의어 못 잡음 → 쿼리 확장(원어 병기) 시도
2. 청크가 잘려 의미 잃음 → 청크 크기·오버랩 조정
3. 다른 책의 비슷한 단어가 우선됨 → 메타데이터 필터(`title`)로 좁히기

### 8-4. 답변 품질

- `SYSTEM_PROMPT` 의 출력 형식 지시를 더 구체적으로
- 추가로 `top_k` 늘리면 컨텍스트 풍부해지지만 잡음도 증가
- 출처 표기가 흐릿하면 컨텍스트 포맷에 페이지·발췌 번호를 더 강조

---

## 9. 다음 단계 후보 (우선순위 제안)

1. **샘플 1권 인제스트 → 검색·질의 검증** ← 가장 먼저
2. OCR 품질 진단 후 정제 규칙 보강
3. 전집 전체 인제스트
4. 평가용 질문 셋 만들기 (몰트만 신학 핵심 질문 20~30개)
5. 다른 신학자 추가 (바르트·판넨베르크·아퀴나스 등 — 메타데이터 YAML만 추가하면 됨)
6. FastAPI 웹 UI
7. 저자 간 비교 질의 (다중 컬렉션 검색 결합)

---

## 10. 알려진 이슈 / 주의

- **클라우드 샌드박스의 git push 가 503으로 자주 실패함** (Anthropic git 프록시 일시 장애).
  맥 로컬에서는 GitHub 직결이라 이런 문제 없음. 그냥 `git push origin <branch>` 사용.
- **저작권**: `data/raw/`, `data/processed/`, `*.pdf` 는 `.gitignore` 처리됨. 절대 GitHub에 올리지 말 것.
- **첫 임베딩 모델 다운로드 ~2GB**: 첫 인제스트는 시간이 걸림. 이후엔 `~/.cache/huggingface` 에 캐시됨.
- **ChromaDB 영속성**: `chroma_db/` 디렉터리에 SQLite + Parquet으로 저장. 백업하려면 이 폴더만 복사.

---

## 11. 빠른 핸드오프 체크리스트

- [ ] 저장소 클론 + 브랜치 체크아웃
- [ ] `uv pip install -e .` 또는 동등
- [ ] `.env` 에 `ANTHROPIC_API_KEY` 설정
- [ ] `data/raw/moltmann/` 에 PDF 배치
- [ ] `data/metadata/moltmann.yaml` 의 `file:` 필드를 실제 파일명과 일치
- [ ] **샘플 1권**으로 `theology-rag ingest moltmann` 시범 실행
- [ ] `theology-rag search` 로 검색 품질 확인
- [ ] OCR 정제 규칙 튜닝 (실제 출력 보고)
- [ ] 전집 인제스트
- [ ] `theology-rag ask` 로 첫 질의

---

## 12. 사용자에게 물어볼 만한 것

핸드오프 받은 직후 사용자에게 확인하면 좋을 항목:

- 보유한 PDF 권 수와 정확한 파일명 (메타데이터 YAML 갱신 위해)
- 번역본인지 원서인지, 역자/출판사 정보 (출처 표기 품질 향상)
- 샘플 페이지를 붙여넣게 해서 OCR 품질을 사용자와 함께 진단
- 우선 관심 주제 (희망의 신학 / 십자가 / 삼위일체 / 종말론 …) — 평가 질문 셋 설계에 사용
