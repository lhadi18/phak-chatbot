from __future__ import annotations

from dataclasses import dataclass

import chromadb

from .config import Settings, load_settings
from .embeddings import encode_texts


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    chapter: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    settings = settings or load_settings()
    collection = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(settings.collection_name)
    available = collection.count()
    if available == 0:
        return []
    vector = encode_texts([query], settings)[0]
    where = {"chapter_number": chapter} if chapter is not None else None
    response = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k or settings.retrieval_top_k, available),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = response["documents"][0]
    metas = response["metadatas"][0]
    distances = response["distances"][0]
    return [RetrievedChunk(text, meta, distance) for text, meta, distance in zip(docs, metas, distances)]


def print_results(results: list[RetrievedChunk]) -> None:
    for rank, item in enumerate(results, start=1):
        meta = item.metadata
        pages = str(meta["page_start"]) if meta["page_start"] == meta["page_end"] else f"{meta['page_start']}-{meta['page_end']}"
        print(f"\n[{rank}] Chapter {meta['chapter_number']} — {meta['chapter_title']} / {meta['section']} / PDF page {pages} (distance {item.distance:.3f})")
        print(item.text)
