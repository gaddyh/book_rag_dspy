import json
from pathlib import Path
from typing import Any

import dspy

from book_rag.core.models import BookChunk


CHUNKS_PATH = Path("data/processed/chunks_unstructured.jsonl")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:  # raw dicts from JSONL
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}. Run ingestion first."
        )

    chunks: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def format_chunk_for_embedding(chunk: dict[str, Any]) -> str:
    """
    Text used for embeddings.

    For unstructured chunks, the text already includes useful context like:
    page, section, element types, etc.
    """
    return chunk["text"]


class BookRetriever:
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embedder_model: str = "openai/text-embedding-3-small",
        k: int = 5,
    ) -> None:
        self.chunks_path = chunks_path
        self.chunks = load_chunks(chunks_path)
        self.k = k

        self.corpus = [
            format_chunk_for_embedding(chunk)
            for chunk in self.chunks
        ]

        self.embedder = dspy.Embedder(embedder_model)

        self.retriever = dspy.retrievers.Embeddings(
            embedder=self.embedder,
            corpus=self.corpus,
            k=k,
        )

    def search(self, question: str, k: int | None = None) -> list[BookChunk]:
        k = k or self.k

        result = self.retriever(question)

        passages = getattr(result, "passages", None)
        indices = getattr(result, "indices", None)

        if passages is None and isinstance(result, dict):
            passages = result.get("passages")
            indices = result.get("indices")

        if passages is None:
            raise TypeError(
                f"Unexpected retriever result type: {type(result)}. "
                f"Value: {result}"
            )

        _known_fields = {"chunk_id", "text", "source", "page_start", "page_end", "has_images", "image_count"}

        matched_chunks: list[BookChunk] = []

        for position, passage in enumerate(passages):
            if indices is not None and position < len(indices):
                corpus_index = indices[position]
                raw = self.chunks[corpus_index]
            else:
                raw = {
                    "chunk_id": "unknown",
                    "text": passage,
                    "source": "unknown",
                    "page_start": None,
                    "page_end": None,
                    "has_images": False,
                    "image_count": 0,
                }

            extra = {key: v for key, v in raw.items() if key not in _known_fields}
            extra["chunks_path"] = str(self.chunks_path)

            matched_chunks.append(
                BookChunk(
                    chunk_id=raw.get("chunk_id", "unknown"),
                    text=raw.get("text", passage),
                    source=raw.get("source", "unknown"),
                    page_start=raw.get("page_start"),
                    page_end=raw.get("page_end"),
                    has_images=raw.get("has_images", False),
                    image_count=raw.get("image_count", 0),
                    rank=position + 1,
                    metadata=extra,
                )
            )

        return matched_chunks