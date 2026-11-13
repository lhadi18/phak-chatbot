from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+(\d+)\b", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+-\d+|\d+\.\d+)(?:\s+|$).{2,120}$")


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class Chunk:
    text: str
    chapter_number: int
    section: str
    page_start: int
    page_end: int


def _is_heading(line: str, font_size: float, max_size: float) -> bool:
    line = " ".join(line.split())
    if len(line) < 3 or len(line) > 140:
        return False
    if _CHAPTER_RE.match(line) or _NUMBERED_HEADING_RE.match(line):
        return True
    words = line.split()
    # Body text is commonly the largest font on a page (especially pages
    # without a title). Treating every short body line as a heading split
    # sentences into tiny sections and poisoned retrieval metadata.
    return (
        font_size >= max_size * 1.15
        and 1 <= len(words) <= 12
        and not line.endswith((".", ",", ";", ":"))
    )


def pages_from_pdf(
    doc: object,
    allowed_chapters: dict[int, str],
    section_ranges: dict[int, tuple[int, int]] | None = None,
) -> list[tuple[PageText, int, str]]:
    """Extract page text and track chapter/section headings from PDF layout."""
    pages: list[tuple[PageText, int, str]] = []
    current_chapter: int | None = None
    current_section = "Unlabeled section"
    section_ranges = section_ranges or {}
    active_section_number: int | None = None

    def section_is_allowed(chapter: int | None) -> bool:
        if chapter not in section_ranges:
            return True
        if active_section_number is None:
            return False
        lower, upper = section_ranges[chapter]
        return lower <= active_section_number <= upper

    for index, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        lines: list[tuple[float, float, str]] = []
        sizes: list[float] = []
        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span["text"] for span in spans).strip()
                if text:
                    size = max((float(span.get("size", 0)) for span in spans), default=0.0)
                    lines.append((float(line["bbox"][1]), size, text))
                    sizes.extend(float(span.get("size", 0)) for span in spans)
        lines.sort(key=lambda item: item[0])
        max_size = max(sizes, default=0.0)
        page_lines: list[str] = []
        local_chapter = current_chapter
        local_section = current_section
        for _, size, text in lines:
            normalized = " ".join(text.split())
            chapter_match = _CHAPTER_RE.match(normalized)
            if chapter_match:
                found = int(chapter_match.group(1))
                if page_lines and local_chapter is not None and section_is_allowed(local_chapter):
                    pages.append((PageText(index + 1, "\n".join(page_lines)), local_chapter, local_section))
                    page_lines = []
                if found in allowed_chapters:
                    current_chapter = found
                    current_section = normalized
                    active_section_number = None
                elif current_chapter is not None:
                    current_chapter = None
                    active_section_number = None
                local_chapter = current_chapter
                local_section = current_section
                continue
            if current_chapter is None:
                continue
            # The PHAK PDFs encode many diagram labels at 6–9 pt. They are
            # interleaved with body copy by geometric text sorting, so omit
            # small unnumbered labels instead of stitching them into prose.
            if size < 9.5 and not _NUMBERED_HEADING_RE.match(normalized):
                continue
            # These are page furniture and are poor retrieval evidence.
            if re.fullmatch(r"\d{1,4}", normalized):
                continue
            if _is_heading(normalized, size, max_size):
                current_section = normalized
                if page_lines and section_is_allowed(local_chapter):
                    pages.append((PageText(index + 1, "\n".join(page_lines)), local_chapter, local_section))
                    page_lines = []
                local_chapter = current_chapter
                local_section = current_section
                section_match = re.match(r"\s*\d+-(\d+)\b", normalized)
                if section_match:
                    active_section_number = int(section_match.group(1))
                if not section_is_allowed(local_chapter):
                    page_lines = []
                    continue
            if not section_is_allowed(current_chapter):
                continue
            page_lines.append(normalized)
        if local_chapter is not None and page_lines and section_is_allowed(local_chapter):
            pages.append((PageText(index + 1, "\n".join(page_lines)), local_chapter, local_section))
    return pages


def make_chunks(
    pages: list[tuple[PageText, int, str]],
    *,
    token_limit: int,
    overlap: int,
) -> list[Chunk]:
    """Group adjacent pages by section, then split oversized sections with overlap."""
    groups: list[tuple[int, str, list[PageText]]] = []
    for page, chapter, section in pages:
        if groups and groups[-1][0] == chapter and groups[-1][1] == section:
            groups[-1][2].append(page)
        else:
            groups.append((chapter, section, [page]))

    result: list[Chunk] = []
    for chapter, section, source_pages in groups:
        pieces: list[str] = []
        page_char_spans: list[tuple[int, int, int]] = []
        char_count = 0
        page_spans: list[tuple[int, int, int]] = []
        for page in source_pages:
            piece = f"[PDF page {page.page_number}]\n{page.text}\n"
            pieces.append(piece)
            page_char_spans.append((char_count, char_count + len(piece), page.page_number))
            char_count += len(piece)
        content = "".join(pieces)
        matches = list(_TOKEN_RE.finditer(content))
        token_index = 0
        for char_start, char_end, page_number in page_char_spans:
            while token_index < len(matches) and matches[token_index].end() <= char_start:
                token_index += 1
            first = token_index
            while token_index < len(matches) and matches[token_index].start() < char_end:
                token_index += 1
            last = token_index
            if first < last:
                page_spans.append((first, last, page_number))
        start = 0
        while start < len(matches):
            end = min(start + token_limit, len(matches))
            text = content[matches[start].start() : matches[end - 1].end()].strip()
            if text:
                page_nums = [number for first, last, number in page_spans if first < end and last > start]
                result.append(Chunk(text, chapter, section, min(page_nums), max(page_nums)))
            if end == len(matches):
                break
            start = max(end - overlap, start + 1)
    return result
