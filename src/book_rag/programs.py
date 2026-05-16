import dspy

from book_rag.retriever import BookRetriever
from book_rag.signatures import AnswerFromBook


class BookRAG(dspy.Module):
    def __init__(self, retriever: BookRetriever, k: int = 5) -> None:
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.answer = dspy.ChainOfThought(AnswerFromBook)

    def forward(self, question: str) -> dspy.Prediction:
        chunks = self.retriever.search(question, k=self.k)

        context_parts = []

        for chunk in chunks:
            page = chunk["page_start"]
            chunk_id = chunk["chunk_id"]
            has_images = chunk.get("has_images", False)
            image_count = chunk.get("image_count", 0)

            context_parts.append(
                f"[{chunk_id} | page {page} | images: {has_images} ({image_count})]\n"
                f"{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prediction = self.answer(
            context=context,
            question=question,
        )

        citations = [
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page_start"],
                "has_images": chunk.get("has_images", False),
                "image_count": chunk.get("image_count", 0),
            }
            for chunk in chunks
        ]

        return dspy.Prediction(
            answer=prediction.answer,
            citations=citations,
            context=context,
            chunks=chunks,
        )