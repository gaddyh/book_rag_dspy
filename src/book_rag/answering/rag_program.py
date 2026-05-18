import dspy

from book_rag.context.context_builder import ChunkOnlyContextBuilder
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
        context_builder=None,
        k: int = 5,
        answer_goal: str = DEFAULT_ANSWER_GOAL,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.context_builder = context_builder or ChunkOnlyContextBuilder()
        self.k = k
        self.answer_goal = answer_goal
        self.answer = dspy.ChainOfThought(AnswerFromBook)

    def forward(self, question: str) -> dspy.Prediction:
        chunks = self.retriever.search(question, k=self.k)
        context = self.context_builder.build(chunks)

        prediction = self.answer(
            context=context.text,
            question=question,
            answer_goal=self.answer_goal,
        )

        citations = [
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_start,
                "page_end": chunk.page_end,
                "has_images": chunk.has_images,
                "image_count": chunk.image_count,
            }
            for chunk in context.chunks
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
            context=context.text,
            chunks=context.chunks,
            answer_goal=self.answer_goal,
        )
