"""쿼리 → 벡터 검색 → top-k 청크 반환."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from theology_rag.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def search(query: str, author: str, chroma_dir: Path, top_k: int = 5) -> list[dict]:
    model = _get_model()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(author)
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
