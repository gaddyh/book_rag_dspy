import sys
from dataclasses import dataclass

from book_rag.retriever import BookRetriever
from book_rag.retriever_metrics import (
    RetrieverMetricConfig,
    RetrieverMetricResult,
    calculate_retriever_metrics,
)


@dataclass
class EvalQuestion:
    id: str
    question: str
    config: RetrieverMetricConfig


QUESTIONS = [
    EvalQuestion(
        id="support_001",
        question="what does the book say about customer support agents?",
        config=RetrieverMetricConfig(
            relevance_keywords=[
                "customer support",
                "support agent",
                "support",
                "ticket",
                "escalat",
                "askai",
                "feedback",
                "customer satisfaction",
            ]
        ),
    ),
    EvalQuestion(
        id="cognitive_loop_001",
        question="what is the cognitive loop?",
        config=RetrieverMetricConfig(
            relevance_keywords=[
                "cognitive loop",
                "perception",
                "reasoning",
                "planning",
                "action",
                "learning",
                "autonomous",
            ]
        ),
    ),
    EvalQuestion(
        id="coding_agent_001",
        question="which parts are relevant to building a coding agent?",
        config=RetrieverMetricConfig(
            relevance_keywords=[
                "software",
                "developer",
                "tester",
                "tests",
                "code",
                "implementation",
                "execution",
                "refinement",
            ]
        ),
    ),
]


K_VALUES = [5, 10, 20]


def print_results(question: str, k: int, results: list[dict]) -> None:
    print("\nQUESTION")
    print("-" * 80)
    print(question)
    print(f"k={k}")

    print("\nTOP MATCHES")
    print("-" * 80)

    for i, chunk in enumerate(results, start=1):
        print(f"\n[{i}] {chunk['chunk_id']}")
        print(f"Page: {chunk['page_start']}")
        print(f"Images: {chunk.get('has_images')} ({chunk.get('image_count')})")
        print()
        print(chunk["text"][:700])
        print("-" * 80)


def print_metrics(metric_result: RetrieverMetricResult) -> None:
    print("\nRETRIEVER METRICS")
    print("-" * 80)

    print(f"k: {metric_result.k}")
    print(f"relevance_to_question: {metric_result.relevance_to_question}")
    print(f"source_quality: {metric_result.source_quality}")
    print(f"final_score: {metric_result.final_score}")

    print()
    print(
        f"relevant chunks: "
        f"{metric_result.relevant_chunk_count}/{metric_result.total_chunks}"
    )
    print(
        f"bad chunks: "
        f"{metric_result.bad_chunk_count}/{metric_result.total_chunks}"
    )

    if metric_result.problems:
        print("\nPROBLEMS")
        print("-" * 80)
        for problem in metric_result.problems:
            print(f"- {problem}")


def print_summary_table(rows: list[dict]) -> None:
    print("\n\nSUMMARY TABLE")
    print("=" * 120)

    headers = [
        "question_id",
        "k",
        "final",
        "relevance",
        "quality",
        "relevant",
        "bad",
        "total",
    ]

    widths = {
        "question_id": 22,
        "k": 5,
        "final": 8,
        "relevance": 10,
        "quality": 10,
        "relevant": 10,
        "bad": 8,
        "total": 8,
    }

    header_line = (
        f"{headers[0]:<{widths['question_id']}}"
        f"{headers[1]:>{widths['k']}}"
        f"{headers[2]:>{widths['final']}}"
        f"{headers[3]:>{widths['relevance']}}"
        f"{headers[4]:>{widths['quality']}}"
        f"{headers[5]:>{widths['relevant']}}"
        f"{headers[6]:>{widths['bad']}}"
        f"{headers[7]:>{widths['total']}}"
    )

    print(header_line)
    print("-" * 120)

    for row in rows:
        print(
            f"{row['question_id']:<{widths['question_id']}}"
            f"{row['k']:>{widths['k']}}"
            f"{row['final_score']:>{widths['final']}.3f}"
            f"{row['relevance_to_question']:>{widths['relevance']}.3f}"
            f"{row['source_quality']:>{widths['quality']}.3f}"
            f"{row['relevant_chunk_count']:>{widths['relevant']}}"
            f"{row['bad_chunk_count']:>{widths['bad']}}"
            f"{row['total_chunks']:>{widths['total']}}"
        )

    print("=" * 120)

    print("\nBEST K PER QUESTION")
    print("-" * 80)

    for question in QUESTIONS:
        question_rows = [
            row for row in rows if row["question_id"] == question.id
        ]
        best_row = max(question_rows, key=lambda row: row["final_score"])

        print(
            f"{question.id}: best k={best_row['k']} "
            f"(final={best_row['final_score']:.3f})"
        )


def run_single_question(question_text: str) -> None:
    config = RetrieverMetricConfig(
        relevance_keywords=question_text.lower().split()
    )

    retriever = BookRetriever(k=5)
    results = retriever.search(question_text)

    metric_result = calculate_retriever_metrics(
        question=question_text,
        chunks=results,
        config=config,
    )

    print_results(question_text, 5, results)
    print_metrics(metric_result)


def run_matrix() -> None:
    summary_rows: list[dict] = []

    for eval_question in QUESTIONS:
        for k in K_VALUES:
            retriever = BookRetriever(k=k)
            results = retriever.search(eval_question.question)

            metric_result = calculate_retriever_metrics(
                question=eval_question.question,
                chunks=results,
                config=eval_question.config,
            )

            print_results(eval_question.question, k, results)
            print_metrics(metric_result)

            summary_rows.append(
                {
                    "question_id": eval_question.id,
                    "question": eval_question.question,
                    "k": k,
                    "final_score": metric_result.final_score,
                    "relevance_to_question": metric_result.relevance_to_question,
                    "source_quality": metric_result.source_quality,
                    "relevant_chunk_count": metric_result.relevant_chunk_count,
                    "bad_chunk_count": metric_result.bad_chunk_count,
                    "total_chunks": metric_result.total_chunks,
                }
            )

    print_summary_table(summary_rows)


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_single_question(question)
        return

    run_matrix()


if __name__ == "__main__":
    main()