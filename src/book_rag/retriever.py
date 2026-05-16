import json
from pathlib import Path
from typing import Any

import dspy


CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}. Run ingest first."
        )

    chunks: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def format_chunk_for_embedding(chunk: dict[str, Any]) -> str:
    return chunk["text"]


class BookRetriever:
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embedder_model: str = "openai/text-embedding-3-small",
        k: int = 5,
    ) -> None:
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

    def search(self, question: str, k: int | None = None) -> list[dict[str, Any]]:
        k = k or self.k

        result = self.retriever(question)

        # DSPy's Embeddings retriever usually returns:
        # Prediction(passages=[...], indices=[...])
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

        matched_chunks: list[dict[str, Any]] = []

        for position, passage in enumerate(passages):
            chunk: dict[str, Any]

            if indices is not None and position < len(indices):
                corpus_index = indices[position]
                chunk = dict(self.chunks[corpus_index])
            else:
                chunk = {
                    "chunk_id": "unknown",
                    "text": passage,
                    "source": "unknown",
                    "page_start": None,
                    "page_end": None,
                    "has_images": False,
                    "image_count": 0,
                }

            chunk["retrieved_text"] = passage
            chunk["rank"] = position + 1

            matched_chunks.append(chunk)

        return matched_chunks