import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf


RAW_PDF_PATH = Path("data/raw/30_agents.pdf")
OUTPUT_PATH = Path("data/processed/chunks_unstructured.jsonl")
ELEMENTS_DEBUG_PATH = Path("data/processed/unstructured_elements_debug.jsonl")


@dataclass
class StructuredChunk:
    chunk_id: str
    text: str
    source: str
    page_start: int | None
    page_end: int | None

    ingestion_method: str
    element_types: list[str]

    section_title: str | None = None
    chapter_title: str | None = None

    has_images: bool = False
    image_count: int = 0

    raw_element_ids: list[str] | None = None


def clean_text(text: str) -> str:
    text = re.sub(r"idx_[a-f0-9]{8}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_element_type(element: Any) -> str:
    return element.__class__.__name__


def get_element_text(element: Any) -> str:
    return clean_text(str(element))


def get_page_number(element: Any) -> int | None:
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return None

    page_number = getattr(metadata, "page_number", None)
    if page_number is None:
        return None

    try:
        return int(page_number)
    except (TypeError, ValueError):
        return None


def get_element_id(element: Any, index: int) -> str:
    element_id = getattr(element, "id", None)
    if element_id:
        return str(element_id)

    return f"element-{index}"


def is_probably_front_matter(page: int | None) -> bool:
    return page is not None and page < 30


def is_noise_element(element_type: str, text: str, page: int | None) -> bool:
    lowered = text.lower()

    if not text:
        return True

    if element_type in {"Header", "Footer"}:
        return True

    if "copyright" in lowered:
        return True

    # Do not keep TOC/front-matter as body chunks for now.
    # Later we may store it separately as navigation metadata.
    if is_probably_front_matter(page):
        if "table of contents" in lowered:
            return True
        if len(text) < 250:
            return True

    return False


def is_title_element(element_type: str) -> bool:
    return element_type in {"Title"}


def is_visual_element(element_type: str) -> bool:
    return element_type in {"Image", "FigureCaption"}


def build_context_text(
    text_parts: list[str],
    chapter_title: str | None,
    section_title: str | None,
    element_types: list[str],
    page_start: int | None,
    page_end: int | None,
) -> str:
    prefix_parts = []

    if chapter_title:
        prefix_parts.append(f"Chapter: {chapter_title}")

    if section_title:
        prefix_parts.append(f"Section: {section_title}")

    if page_start is not None:
        if page_end is not None and page_end != page_start:
            prefix_parts.append(f"Pages: {page_start}-{page_end}")
        else:
            prefix_parts.append(f"Page: {page_start}")

    if element_types:
        prefix_parts.append(f"Element types: {', '.join(sorted(set(element_types)))}")

    prefix = "\n".join(prefix_parts)
    body = "\n\n".join(text_parts)

    if prefix:
        return f"{prefix}\n\n{body}"

    return body


def chunk_elements(
    elements: list[Any],
    source_name: str,
    chunk_size: int = 1600,
    overlap_elements: int = 1,
) -> list[StructuredChunk]:
    """
    Chunk by semantic elements, not raw characters.

    Accumulate elements until chunk_size is reached. Keep current section title.
    """
    chunks: list[StructuredChunk] = []

    current_section_title: str | None = None
    current_chapter_title: str | None = None

    buffer_texts: list[str] = []
    buffer_element_types: list[str] = []
    buffer_pages: list[int] = []
    buffer_ids: list[str] = []
    buffer_has_images = False
    buffer_image_count = 0

    previous_kept_elements: list[tuple[str, str, int | None, str]] = []

    def flush() -> None:
        nonlocal buffer_texts
        nonlocal buffer_element_types
        nonlocal buffer_pages
        nonlocal buffer_ids
        nonlocal buffer_has_images
        nonlocal buffer_image_count

        if not buffer_texts:
            return

        page_start = min(buffer_pages) if buffer_pages else None
        page_end = max(buffer_pages) if buffer_pages else None

        context_text = build_context_text(
            text_parts=buffer_texts,
            chapter_title=current_chapter_title,
            section_title=current_section_title,
            element_types=buffer_element_types,
            page_start=page_start,
            page_end=page_end,
        )

        chunk_index = len(chunks)

        chunk = StructuredChunk(
            chunk_id=f"{source_name}:unstructured:chunk-{chunk_index}",
            text=context_text,
            source=source_name,
            page_start=page_start,
            page_end=page_end,
            ingestion_method="unstructured_fast",
            element_types=sorted(set(buffer_element_types)),
            chapter_title=current_chapter_title,
            section_title=current_section_title,
            has_images=buffer_has_images,
            image_count=buffer_image_count,
            raw_element_ids=list(buffer_ids),
        )

        chunks.append(chunk)

        # Element-level overlap.
        if overlap_elements > 0:
            overlap = previous_kept_elements[-overlap_elements:]
            buffer_texts = [item[0] for item in overlap]
            buffer_element_types = [item[1] for item in overlap]
            buffer_pages = [item[2] for item in overlap if item[2] is not None]
            buffer_ids = [item[3] for item in overlap]
            buffer_has_images = any(item[1] in {"Image", "FigureCaption"} for item in overlap)
            buffer_image_count = sum(1 for item in overlap if item[1] in {"Image", "FigureCaption"})
        else:
            buffer_texts = []
            buffer_element_types = []
            buffer_pages = []
            buffer_ids = []
            buffer_has_images = False
            buffer_image_count = 0

    for index, element in enumerate(elements):
        element_type = get_element_type(element)
        text = get_element_text(element)
        page = get_page_number(element)
        element_id = get_element_id(element, index)

        if is_noise_element(element_type, text, page):
            continue

        if is_title_element(element_type):
            # Simple heuristic:
            # Longer/chapter-looking titles become chapter-ish.
            # Shorter body titles become section titles.
            lowered = text.lower()

            if lowered.startswith("chapter ") or re.match(r"chapter\s+\d+", lowered):
                current_chapter_title = text
                current_section_title = None
            else:
                current_section_title = text

        if is_visual_element(element_type):
            buffer_has_images = True
            buffer_image_count += 1

        candidate_len = sum(len(part) for part in buffer_texts) + len(text)

        if buffer_texts and candidate_len > chunk_size:
            flush()

        buffer_texts.append(text)
        buffer_element_types.append(element_type)

        if page is not None:
            buffer_pages.append(page)

        buffer_ids.append(element_id)
        previous_kept_elements.append((text, element_type, page, element_id))

    flush()

    return chunks


def save_chunks(chunks: list[StructuredChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def save_debug_elements(elements: list[Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for index, element in enumerate(elements):
            payload = {
                "index": index,
                "id": get_element_id(element, index),
                "type": get_element_type(element),
                "page": get_page_number(element),
                "text": get_element_text(element)[:1000],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def print_element_summary(elements: list[Any]) -> None:
    type_counts = Counter(get_element_type(element) for element in elements)
    pages = [
        get_page_number(element)
        for element in elements
        if get_page_number(element) is not None
    ]

    print("\nUNSTRUCTURED ELEMENT SUMMARY")
    print("-" * 80)
    print(f"Elements: {len(elements)}")
    print(f"Pages with metadata: {len(set(pages))}")

    print("\nElement type counts:")
    for element_type, count in type_counts.most_common():
        print(f"- {element_type}: {count}")


def main() -> None:
    if not RAW_PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {RAW_PDF_PATH.resolve()}")

    print(f"Partitioning PDF with Unstructured: {RAW_PDF_PATH.resolve()}")

    elements = partition_pdf(
        filename=str(RAW_PDF_PATH),
        strategy="fast",
    )

    print_element_summary(elements)
    save_debug_elements(elements, ELEMENTS_DEBUG_PATH)

    chunks = chunk_elements(
        elements=elements,
        source_name=RAW_PDF_PATH.name,
    )

    save_chunks(chunks, OUTPUT_PATH)

    print("\nUNSTRUCTURED CHUNKS")
    print("-" * 80)
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")
    print(f"Debug elements saved to: {ELEMENTS_DEBUG_PATH.resolve()}")

    if chunks:
        print("\nFirst chunk preview:")
        print("-" * 80)
        print(chunks[0].text[:1200])
        print("-" * 80)
        print(chunks[0])


if __name__ == "__main__":
    main()