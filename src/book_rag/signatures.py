import dspy


class AnswerFromBook(dspy.Signature):
    """
    Answer the user's question using only the provided context from the book.

    Use the retrieved context as evidence.
    If the context is not enough to answer, say that the retrieved context
    is insufficient.

    Prefer a useful learning answer:
    - direct answer
    - key points
    - why it matters
    """

    context: str = dspy.InputField(
        desc="Relevant retrieved passages from the book, including source/page metadata."
    )
    question: str = dspy.InputField(desc="User question about the book.")

    answer: str = dspy.OutputField(
        desc=(
            "A grounded answer based only on the provided context. "
            "Include a concise definition, key stages or components if relevant, "
            "and why it matters."
        )
    )