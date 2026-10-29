from __future__ import annotations

from dataclasses import dataclass
import re

from .config import Settings, load_settings
from .prompts import ASK_SYSTEM, ASK_USER
from .providers import LLMProvider, create_provider
from .retrieval import RetrievedChunk, retrieve


@dataclass
class AskResult:
    answer: str
    sources: list[RetrievedChunk]
    citation_warning: str | None = None


def _context(sources: list[RetrievedChunk]) -> str:
    entries = []
    for index, item in enumerate(sources, 1):
        meta = item.metadata
        page_range = str(meta["page_start"]) if meta["page_start"] == meta["page_end"] else f"{meta['page_start']}-{meta['page_end']}"
        entries.append(
            f"[S{index}] Chapter {meta['chapter_number']}: {meta['chapter_title']} | "
            f"Section: {meta['section']} | PDF page(s): {page_range}\n{item.text}"
        )
    return "\n\n---\n\n".join(entries)


def _citation_warning(answer: str, source_count: int) -> str | None:
    labels = [int(label) for label in re.findall(r"\[S(\d+)\]", answer)]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]
    if not labels or any(label < 1 or label > source_count for label in labels):
        return "Citation check failed: the answer has missing or invalid source labels; verify it against the excerpts."
    if any(not re.search(r"\[S\d+\]", sentence) for sentence in sentences):
        return "Citation check: one or more sentences lack a source label; verify those claims against the excerpts."
    return None


def ask(
    question: str,
    *,
    chapter: int | None = None,
    top_k: int | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    on_token=None,
) -> AskResult:
    settings = settings or load_settings()
    sources = retrieve(question, chapter=chapter, top_k=top_k, settings=settings)
    if not sources:
        return AskResult("No matching source chunks were found. Ingest the PHAK PDFs first.", [])
    provider = provider or create_provider(settings)
    prompt = ASK_USER.format(question=question, context=_context(sources))
    answer_parts = []
    for fragment in provider.stream_text(prompt, system=ASK_SYSTEM):
        answer_parts.append(fragment)
        if on_token:
            on_token(fragment)
    answer = "".join(answer_parts)
    return AskResult(answer, sources, _citation_warning(answer, len(sources)))


def print_source_list(sources: list[RetrievedChunk]) -> None:
    print("\nRetrieved excerpts:")
    for index, item in enumerate(sources, 1):
        meta = item.metadata
        pages = str(meta["page_start"]) if meta["page_start"] == meta["page_end"] else f"{meta['page_start']}-{meta['page_end']}"
        print(f"\n[S{index}] Chapter {meta['chapter_number']}: {meta['chapter_title']} — {meta['section']} — PDF page(s) {pages}")
        print(item.text)
