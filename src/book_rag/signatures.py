import dspy


class AnswerFromBook(dspy.Signature):
    """
    Answer the user's question using only the provided context from the book.

    Goal:
    Help the user learn the book as an AI engineer building real systems.

    Requirements:
    - Stay grounded in the retrieved context.
    - If the context is insufficient, say so clearly.
    - Separate what the book says from practical interpretation.
    - Prefer practical design, implementation, debugging, and evaluation takeaways.
    - Do not turn a specific example into a universal rule.
    """

    context: str = dspy.InputField(
        desc=(
            "Relevant retrieved passages from the book, including source/page metadata. "
            "Use this as the only source of truth."
        )
    )

    question: str = dspy.InputField(
        desc="User question about the book."
    )

    answer_goal: str = dspy.InputField(
        desc=(
            "The learning goal for the answer. Example: explain the book content "
            "in a practical, AI-engineering-oriented way."
        )
    )

    direct_answer: str = dspy.OutputField(
        desc=(
            "A short direct answer to the user's question. "
            "No long explanation yet."
        )
    )

    book_summary: str = dspy.OutputField(
        desc=(
            "What the retrieved book context says. "
            "This should be grounded only in the context."
        )
    )

    key_points: list[str] = dspy.OutputField(
        desc=(
            "Important grounded points from the retrieved context. "
            "Use short bullet-style strings."
        )
    )

    practical_usage: str = dspy.OutputField(
        desc=(
            "How the user can apply this idea when designing, building, debugging, "
            "or evaluating AI agent/RAG systems. Clearly mark this as practical "
            "interpretation, not necessarily a direct quote from the book."
        )
    )

    design_takeaway: str = dspy.OutputField(
        desc=(
            "The main engineering/design lesson the user should take away."
        )
    )

    caveats: list[str] = dspy.OutputField(
        desc=(
            "Caveats, limits, or warnings. Include if context is insufficient, "
            "if an example should not be generalized, or if images/figures were present "
            "but not interpreted."
        )
    )

    answer: str = dspy.OutputField(
        desc=(
            "Final user-facing answer combining the fields above into a clear, "
            "practical learning answer. Structure it with short sections."
        )
    )