"""PDF → 정제 → 청크 → 임베딩 → ChromaDB 저장."""
from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from theology_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    try:
        return [(i + 1, doc[i].get_text()) for i in range(len(doc))]
    finally:
        doc.close()


def clean_ocr_text(text: str) -> str:
    text = re.sub(r"([A-Za-zäöüÄÖÜß])-\n([A-Za-zäöüÄÖÜß])", r"\1\2", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    total_added = 0
    for work in works:
        pdf_name = work["file"]
        pdf_path = raw_dir / pdf_name
        if not pdf_path.exists():
            print(f"  [skip] {pdf_name} (파일 없음: {pdf_path})")
            continue

        print(f"\n처리 중: {pdf_name}")
        pages = extract_pages(pdf_path)

        chunks_out: list[dict] = []
        for page_num, page_text in pages:
            cleaned = clean_ocr_text(page_text)
            if not cleaned:
                continue
            for chunk_idx, chunk in enumerate(chunk_text(cleaned)):
                chunks_out.append({
                    "text": chunk,
                    "author": author,
                    "title": work.get("title", pdf_name),
                    "year": work.get("year"),
                    "page": page_num,
                    "file": pdf_name,
                    "chunk_idx": chunk_idx,
                })

        if not chunks_out:
            print(f"  [warn] 추출된 청크 없음 — OCR 품질을 확인하세요.")
            continue

        jsonl_path = processed_dir / f"{Path(pdf_name).stem}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in chunks_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        texts = [c["text"] for c in chunks_out]
        print(f"  임베딩 {len(texts)}개 생성 중...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=16).tolist()
        ids = [f"{author}:{Path(pdf_name).stem}:{c['chunk_idx']}:p{c['page']}" for c in chunks_out]
        metadatas = [{k: v for k, v in c.items() if k != "text" and v is not None} for c in chunks_out]
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        total_added += len(chunks_out)
        print(f"  ✓ {len(chunks_out)}개 청크 업서트")

    print(f"\n완료: {author} 컬렉션에 총 {total_added}개 청크 처리됨")
