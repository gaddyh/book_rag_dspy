import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source: str
    page_start: int
    page_end: int
    has_images: bool = False
    image_count: int = 0


def clean_pdf_text(text: str) -> str:
    """
    Remove common PDF extraction artifacts and normalize whitespace.
    """
    text = re.sub(r"idx_[a-f0-9]{8}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    page_number: int,
    chunk_size: int = 1400,
    overlap: int = 200,
    has_images: bool = False,
    image_count: int = 0,
) -> list[TextChunk]:
    """
    Split one page of text into overlapping chunks.

    We extract page by page, then chunk within each page.
    This keeps citations simple because every chunk still belongs
    to a known PDF page.
    """
    cleaned = clean_pdf_text(text)

    if not cleaned:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end].strip()

        if chunk:
            chunks.append(
                TextChunk(
                    chunk_id=f"{source}:page-{page_number}:chunk-{index}",
                    text=chunk,
                    source=source,
                    page_start=page_number,
                    page_end=page_number,
                    has_images=has_images,
                    image_count=image_count,
                )
            )

        start += chunk_size - overlap
        index += 1

    return chunks