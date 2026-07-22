"""쿼리 → 벡터 검색 → top-k 청크 반환."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from symposium.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_client(chroma_dir: str):
    """PersistentClient 재사용 — 호출당 재생성 방지(sqlite 핸들·컬렉션 메타 재로드 절감).

    재인덱싱(ingest)은 별도 프로세스라 서버 프로세스의 캐시와 무관하다.
    운영 절차: 컬렉션 재인덱싱 후에는 서버를 재시작해야 새 데이터가 반영된다.
    """
    return chromadb.PersistentClient(path=chroma_dir)


@lru_cache(maxsize=64)
def _get_collection(chroma_dir: str, author: str):
    return _get_client(chroma_dir).get_collection(author)


def search(query: str, author: str, chroma_dir: Path, top_k: int = 5) -> list[dict]:
    model = _get_model()
    collection = _get_collection(str(chroma_dir), author)
    query_embedding = model.encode([query]).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits: list[dict] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for i in range(len(ids)):
        hits.append({"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]})
    return hits
