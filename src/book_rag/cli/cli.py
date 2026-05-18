import os
import sys

import dspy
from dotenv import load_dotenv

from book_rag.answering.rag_program import BookRAG
from book_rag.context.context_builder import ChunkOnlyContextBuilder
from book_rag.retriever.retriever import BookRetriever


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


def print_list(title: str, values) -> None:
    print(f"\n{title}")
    print("-" * 80)

    if not values:
        print("-")
        return

    if isinstance(values, str):
        print(values)
        return

    for item in values:
        print(f"- {item}")


def print_answer(question: str, prediction) -> None:
    print("\nQUESTION")
    print("-" * 80)
    print(question)

    print("\nDIRECT ANSWER")
    print("-" * 80)
    print(prediction.direct_answer)

    print("\nWHAT THE BOOK SAYS")
    print("-" * 80)
    print(prediction.book_summary)

    print_list("KEY POINTS", prediction.key_points)

    print("\nPRACTICAL USAGE")
    print("-" * 80)
    print(prediction.practical_usage)

    print("\nDESIGN TAKEAWAY")
    print("-" * 80)
    print(prediction.design_takeaway)

    print_list("CAVEATS", prediction.caveats)

    print("\nFINAL ANSWER")
    print("-" * 80)
    print(prediction.answer)

    print("\nRETRIEVED SOURCES")
    print("-" * 80)

    for citation in prediction.citations:
        image_note = ""
        if citation["has_images"]:
            image_note = f" | page has {citation['image_count']} image(s)"

        page = citation.get("page")
        page_end = citation.get("page_end")

        if page is not None and page_end is not None and page_end != page:
            page_label = f"pages {page}-{page_end}"
        elif page is not None:
            page_label = f"page {page}"
        else:
            page_label = "page unknown"

        print(
            f"- {citation['chunk_id']} | {page_label}{image_note}"
        )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: PYTHONPATH=src python -m book_rag.cli "your question"'
        )

    question = " ".join(sys.argv[1:])

    configure_dspy()

    retriever = BookRetriever(k=5)
    rag = BookRAG(
        retriever=retriever,
        context_builder=ChunkOnlyContextBuilder(),
        k=5,
    )

    prediction = rag(question=question)

    print_answer(question, prediction)


if __name__ == "__main__":
    main()