import dspy

from book_rag.retriever.retriever import BookRetriever
from book_rag.signatures import AnswerFromBook


DEFAULT_ANSWER_GOAL = (
    "Help the user learn the book as an AI engineer building real systems. "
    "Explain concepts in a practical, implementation-oriented way. "
    "Prefer design patterns, architecture implications, evaluation ideas, "
    "and concrete usage. Stay grounded in the retrieved context. "
    "Clearly separate book content from practical interpretation."
)


class BookRAG(dspy.Module):
    def __init__(
        self,
        retriever: BookRetriever,
        k: int = 5,
        answer_goal: str = DEFAULT_ANSWER_GOAL,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.answer_goal = answer_goal
        self.answer = dspy.ChainOfThought(AnswerFromBook)

    def forward(self, question: str) -> dspy.Prediction:
        chunks = self.retriever.search(question, k=self.k)

        context_parts = []

        for chunk in chunks:
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
            chunk_id = chunk.get("chunk_id")
            has_images = chunk.get("has_images", False)
            image_count = chunk.get("image_count", 0)

            if page_start is not None and page_end is not None and page_end != page_start:
                page_label = f"pages {page_start}-{page_end}"
            elif page_start is not None:
                page_label = f"page {page_start}"
            else:
                page_label = "page unknown"

            context_parts.append(
                f"[{chunk_id} | {page_label} | images: {has_images} ({image_count})]\n"
                f"{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prediction = self.answer(
            context=context,
            question=question,
            answer_goal=self.answer_goal,
        )

        citations = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "has_images": chunk.get("has_images", False),
                "image_count": chunk.get("image_count", 0),
            }
            for chunk in chunks
        ]

        return dspy.Prediction(
            answer=prediction.answer,
            direct_answer=prediction.direct_answer,
            book_summary=prediction.book_summary,
            key_points=prediction.key_points,
            practical_usage=prediction.practical_usage,
            design_takeaway=prediction.design_takeaway,
            caveats=prediction.caveats,
            citations=citations,
            context=context,
            chunks=chunks,
            answer_goal=self.answer_goal,
        )