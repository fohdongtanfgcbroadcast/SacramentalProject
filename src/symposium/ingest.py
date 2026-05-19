"""PDF/TXT/HTML → 정제 → 청크 → 임베딩 → ChromaDB 저장."""
from __future__ import annotations

import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from symposium.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from symposium.textclean import strip_web_chrome


# --- 텍스트 추출 ---

def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF — PDF일 때만 import
    doc = fitz.open(pdf_path)
    try:
        return [(i + 1, doc[i].get_text()) for i in range(len(doc))]
    finally:
        doc.close()


class _HTMLStripper(HTMLParser):
    """HTML 태그를 제거하고 텍스트만 추출."""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True
        elif tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def extract_text_file(path: Path) -> str:
    """TXT 또는 HTML 파일에서 텍스트 추출."""
    content = path.read_text(encoding="utf-8", errors="replace")
    # 확장자가 .txt 라도 내용이 HTML 이면 태그 제거 (잘못 저장된 웹 페이지 대응)
    head = content[:2000].lower()
    is_html = path.suffix.lower() in (".html", ".htm") or (
        "<!doctype html" in head or "<html" in head
        or ("<meta " in head and "<title" in head)
    )
    if is_html:
        stripper = _HTMLStripper()
        stripper.feed(content)
        return stripper.get_text()
    return content


# --- 정제 ---

def clean_text(text: str) -> str:
    """범용 텍스트 정제. PDF OCR과 플레인텍스트 모두 대응."""
    # 하이픈 줄바꿈 병합 (영어/독일어)
    text = re.sub(r"([A-Za-zäöüÄÖÜß])-\n([A-Za-zäöüÄÖÜß])", r"\1\2", text)
    # 단독 페이지 번호 제거
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Gutenberg 머리글/꼬리글
    text = re.sub(r"^\*\*\* ?(START|END) OF TH.+\*\*\*.*$", "", text, flags=re.MULTILINE)
    # 러닝 헤더/풋터 (한국어 PDF용)
    text = re.sub(r"^\s*\d{1,4}\s+.{2,30}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*제\d+장\s+.{2,30}\s+\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*서론[:：]?\s+.{2,30}\s+\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 본문 시작 전 스킵할 페이지 수 (표지, 속표지, 판권 등) — PDF 전용
SKIP_FRONT_PAGES = 5
# 최소 청크 길이 (자) — 이보다 짧으면 노이즈로 간주
MIN_CHUNK_LENGTH = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            para = text.rfind("\n\n", start, end)
            if para > start + chunk_size // 2:
                end = para + 2
            else:
                sent = max(
                    text.rfind(". ", start, end),
                    text.rfind("。", start, end),
                    text.rfind("다. ", start, end),
                )
                if sent > start + chunk_size // 2:
                    end = sent + 2
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# --- 인제스트 ---

def _extract_work(file_path: Path, is_pdf: bool) -> list[tuple[int, str]]:
    """파일 형식에 따라 텍스트 추출. PDF는 페이지 단위, TXT/HTML은 단일 블록."""
    if is_pdf:
        return extract_pages(file_path)
    else:
        full_text = extract_text_file(file_path)
        # 소스 chrome/boilerplate 제거 (Gutenberg START/END·CCEL·IA 등).
        # 가상 페이지 분할 전에 적용 — 슬라이싱이 원본 라인 구조에 의존.
        full_text = strip_web_chrome(full_text)
        # 텍스트를 가상 페이지로 분할 (약 3000자 단위)
        page_size = 3000
        pages = []
        for i in range(0, len(full_text), page_size):
            page_num = i // page_size + 1
            pages.append((page_num, full_text[i:i + page_size]))
        return pages


def ingest_author(author: str, data_root: Path, metadata_path: Path, chroma_dir: Path) -> None:
    with open(metadata_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    works = meta.get("works", [])
    if not works:
        raise SystemExit(f"{metadata_path}에 works 항목이 없습니다.")

    raw_dir = data_root / "raw" / author
    processed_dir = data_root / "processed" / author
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"임베딩 모델 로딩 중: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(author, metadata={"hnsw:space": "cosine"})

    # macOS APFS/HFS+ 유니코드 정규화(NFD) 대응: 파일명 인덱스 구축
    _fs_files: dict[str, Path] = {}
    if raw_dir.exists():
        for p in raw_dir.iterdir():
            _fs_files[unicodedata.normalize("NFC", p.name)] = p

    total_added = 0
    for work in works:
        file_name = work["file"]
        file_path = _fs_files.get(file_name) or raw_dir / file_name
        if not file_path.exists():
            print(f"  [skip] {file_name} (파일 없음)")
            continue

        is_pdf = file_path.suffix.lower() == ".pdf"
        skip_pages = SKIP_FRONT_PAGES if is_pdf else 0

        print(f"\n처리 중: {file_name}")
        pages = _extract_work(file_path, is_pdf)

        chunks_out: list[dict] = []
        for page_num, page_text in pages:
            if page_num <= skip_pages:
                continue
            cleaned = clean_text(page_text)
            if not cleaned or len(cleaned) < MIN_CHUNK_LENGTH:
                continue
            for chunk_idx, chunk in enumerate(chunk_text(cleaned)):
                if len(chunk) < MIN_CHUNK_LENGTH:
                    continue
                chunks_out.append({
                    "text": chunk,
                    "author": author,
                    "title": work.get("title", file_name),
                    "year": work.get("year"),
                    "page": page_num,
                    "file": file_name,
                    "chunk_idx": chunk_idx,
                })

        if not chunks_out:
            print(f"  [warn] 추출된 청크 없음")
            continue

        jsonl_path = processed_dir / f"{Path(file_name).stem}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in chunks_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        texts = [c["text"] for c in chunks_out]
        print(f"  임베딩 {len(texts)}개 생성 중...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=16).tolist()
        ids = [f"{author}:{Path(file_name).stem}:{c['chunk_idx']}:p{c['page']}" for c in chunks_out]
        metadatas = [{k: v for k, v in c.items() if k != "text" and v is not None} for c in chunks_out]
        # ChromaDB 최대 배치 크기 제한 대응
        BATCH_LIMIT = 5000
        for b_start in range(0, len(ids), BATCH_LIMIT):
            b_end = b_start + BATCH_LIMIT
            collection.upsert(
                ids=ids[b_start:b_end],
                embeddings=embeddings[b_start:b_end],
                documents=texts[b_start:b_end],
                metadatas=metadatas[b_start:b_end],
            )
        total_added += len(chunks_out)
        print(f"  ✓ {len(chunks_out)}개 청크 업서트")

    print(f"\n완료: {author} 컬렉션에 총 {total_added}개 청크 처리됨")
