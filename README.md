# SecramentalProject — 신학 문헌 RAG

**현대·교부 신학자의 저작을 벡터 DB에 넣고 Claude로 질의응답하는 RAG 시스템.**
첫 번째 대상: 위르겐 몰트만(Jürgen Moltmann) 전집 (OCR PDF).

## 구성 요소

- **LLM**: Claude Opus 4.7 (프롬프트 캐싱 적용)
- **벡터 DB**: ChromaDB (로컬 영속)
- **임베딩**: `BAAI/bge-m3` (다국어, 한/영/독 동시 처리)
- **PDF 추출**: PyMuPDF + 간단한 OCR 정제 규칙
- **CLI**: Typer + Rich

## 디렉터리 구조

```
SecramentalProject/
├── data/
│   ├── raw/                  # PDF 원본 (gitignore, 로컬에만 보관)
│   │   └── moltmann/
│   ├── processed/            # 정제된 JSONL (gitignore, 재생성 가능)
│   └── metadata/
│       └── moltmann.yaml     # 저작 목록·서지 메타데이터 (커밋)
├── chroma_db/                # 벡터 DB (gitignore, 재생성 가능)
├── src/theology_rag/
│   ├── config.py
│   ├── ingest.py             # PDF → 정제 → 청크 → 임베딩 → Chroma
│   ├── retrieve.py           # 쿼리 → top-k 발췌
│   ├── generate.py           # Claude API (프롬프트 캐싱 적용)
│   └── cli.py                # typer CLI
├── pyproject.toml
└── .env.example
```

## 설치

Python 3.11+ 필요. [uv](https://github.com/astral-sh/uv) 권장:

```bash
uv venv
uv pip install -e .
cp .env.example .env  # ANTHROPIC_API_KEY 채우기
```

또는 pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## 몰트만 인제스트 (기본 흐름)

1. PDF를 `data/raw/moltmann/` 아래에 둔다.
2. `data/metadata/moltmann.yaml` 의 `file:` 항목을 실제 PDF 파일명과 맞춘다.
3. 인제스트 실행:

```bash
theology-rag ingest moltmann
```

> 첫 실행 시 `BAAI/bge-m3` 모델(≈2GB)이 Hugging Face에서 다운로드됩니다.

## 사용

### 검색만 (Claude 호출 없음)

```bash
theology-rag search "십자가 신학" --author moltmann --top-k 5
```

### 검색 + Claude 답변

```bash
theology-rag ask "몰트만의 종말론에서 '희망'은 어떻게 정의되는가?"
```

## 저작권 주의

`data/raw/` 와 `data/processed/` 는 `.gitignore` 에 포함되어 있습니다.
**저작권 있는 본문·번역본은 절대 GitHub에 커밋하지 마세요.**
서지 메타데이터(제목·저자·출판년도)만 리포지토리에 유지됩니다.
