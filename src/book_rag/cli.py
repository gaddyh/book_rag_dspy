import os
import sys

import dspy
from dotenv import load_dotenv

from book_rag.programs import BookRAG
from book_rag.retriever import BookRetriever


def configure_dspy() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or export it in your shell."
        )

    lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_key=api_key,
    )

    dspy.configure(lm=lm)


def print_answer(question: str, prediction) -> None:
    print("\nQUESTION")
    print("-" * 80)
    print(question)

    print("\nANSWER")
    print("-" * 80)
    print(prediction.answer)

    print("\nRETRIEVED SOURCES")
    print("-" * 80)

    for citation in prediction.citations:
        image_note = ""
        if citation["has_images"]:
            image_note = f" | page has {citation['image_count']} image(s)"

        print(
            f"- {citation['chunk_id']} | page {citation['page']}{image_note}"
        )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: PYTHONPATH=src python -m book_rag.cli "your question"'
        )

    question = " ".join(sys.argv[1:])

    configure_dspy()

    retriever = BookRetriever(k=5)
    rag = BookRAG(retriever=retriever, k=5)

    prediction = rag(question=question)

    print_answer(question, prediction)


if __name__ == "__main__":
    main()