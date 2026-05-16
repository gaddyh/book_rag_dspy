import dspy


class AnswerFromBook(dspy.Signature):
    """
    Answer the user's question using only the provided context from the book.

    If the context is not enough to answer, say that the retrieved context
    is insufficient.
    """

    context: str = dspy.InputField(
        desc="Relevant retrieved passages from the book, including source/page metadata."
    )
    question: str = dspy.InputField(desc="User question about the book.")

    answer: str = dspy.OutputField(
        desc="Grounded answer based only on the provided context."
    )