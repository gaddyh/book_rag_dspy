import sys

from book_rag.retriever import BookRetriever


def print_results(question: str, results: list[dict]) -> None:
    print("\nQUESTION")
    print("-" * 80)
    print(question)

    print("\nTOP MATCHES")
    print("-" * 80)

    for i, chunk in enumerate(results, start=1):
        print(f"\n[{i}] {chunk['chunk_id']}")
        print(f"Page: {chunk['page_start']}")
        print(f"Score: {chunk.get('score')}")
        print(f"Images: {chunk.get('has_images')} ({chunk.get('image_count')})")
        print()
        print(chunk["text"][:900])
        print("-" * 80)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: PYTHONPATH=src python -m book_rag.cli_retrieve "your question"'
        )

    question = " ".join(sys.argv[1:])

    retriever = BookRetriever(k=5)
    results = retriever.search(question, k=5)

    print_results(question, results)


if __name__ == "__main__":
    main()