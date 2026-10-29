from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
import fitz

from .chunking import make_chunks, pages_from_pdf
from .config import Settings, load_settings
from .embeddings import encode_texts


def _chunk_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}\0{chunk_index}\0{text}".encode()).hexdigest()
    return digest


def ingest(settings: Settings | None = None) -> int:
    settings = settings or load_settings()
    pdfs = sorted(settings.data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {settings.data_dir}. Add the PHAK PDF, then rerun ingest.")
    if not settings.chapters:
        raise ValueError("No chapters configured in config.yaml.")

    chroma = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = chroma.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
    )
    total = 0
    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        extracted = pages_from_pdf(doc, settings.chapters, settings.chapter_section_ranges)
        chunks = make_chunks(
            extracted,
            token_limit=settings.chunk_tokens,
            overlap=settings.chunk_overlap_tokens,
        )
        # Replacing all records for this source makes repeat ingestion deterministic and
        # removes stale chunks if the PDF or chunking configuration changed. Embed first
        # so an API failure leaves the prior local index intact.
        embedded_batches = []
        for offset in range(0, len(chunks), settings.embedding_batch_size):
            batch = chunks[offset : offset + settings.embedding_batch_size]
            embeddings = encode_texts([chunk.text for chunk in batch], settings)
            embedded_batches.append((offset, batch, embeddings))
        collection.delete(where={"source_file": pdf_path.name})
        for offset, batch, embeddings in embedded_batches:
            ids = [_chunk_id(pdf_path.name, offset + i, chunk.text) for i, chunk in enumerate(batch)]
            collection.add(
                ids=ids,
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings,
                metadatas=[
                    {
                        "source_file": pdf_path.name,
                        "chapter_number": chunk.chapter_number,
                        "chapter_title": settings.chapters[chunk.chapter_number],
                        "section": chunk.section or "Unlabeled section",
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    }
                    for chunk in batch
                ],
            )
        total += len(chunks)
        print(f"{pdf_path.name}: indexed {len(chunks)} chunks from {len(extracted)} in-scope pages")
    print(f"Done. Indexed {total} chunks across {len(pdfs)} PDF(s).")
    return total
