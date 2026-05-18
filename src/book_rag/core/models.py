from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BookChunk:
    chunk_id: str
    text: str
    source: str
    page_start: int | None = None
    page_end: int | None = None
    has_images: bool = False
    image_count: int = 0
    rank: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    chunks: list[BookChunk]
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)
