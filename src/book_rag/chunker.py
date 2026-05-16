from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source: str
    page_start: int
    page_end: int


def chunk_text(
    text: str,
    source: str,
    page_number: int,
    chunk_size: int = 1400,
    overlap: int = 200,
) -> list[TextChunk]:
    """
    Split one page of text into overlapping chunks.

    We keep page metadata because later the RAG answer must cite where
    the answer came from.
    """
    cleaned = " ".join(text.split())

    if not cleaned:
        return []

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
                )
            )

        start += chunk_size - overlap
        index += 1

    return chunks
