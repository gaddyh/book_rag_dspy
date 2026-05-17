from dataclasses import dataclass
from pathlib import Path

import dspy

from book_rag.retriever.retriever import BookRetriever
from book_rag.retriever.retriever_metrics import RetrieverMetricConfig
from book_rag.retriever.retriever_metrics_llm import (
    LLMRetrieverMetricResult,
    JudgeRetrievedChunk,
    calculate_llm_retriever_metrics,
    configure_dspy_for_metrics,
)


@dataclass
class EvalQuestion:
    id: str
    question: str
    config: RetrieverMetricConfig


CHUNKS_PATH = Path("data/processed/chunks_unstructured.jsonl")


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


def print_llm_metrics(result: LLMRetrieverMetricResult) -> None:
    print("\nLLM RETRIEVER METRICS")
    print("-" * 80)
    print(f"k: {result.k}")
    print(f"final_score: {result.final_score}")
    print(f"average_semantic_relevance: {result.average_semantic_relevance}")
    print(f"average_evidence_value: {result.average_evidence_value}")
    print(f"average_source_quality: {result.average_source_quality}")
    print(f"context_efficiency: {result.context_efficiency}")
    print()
    print(
        f"central/supporting/background/irrelevant/junk: "
        f"{result.central_count}/"
        f"{result.supporting_count}/"
        f"{result.background_count}/"
        f"{result.irrelevant_count}/"
        f"{result.junk_count}"
    )
    print(f"useful chunks: {result.useful_chunk_count}/{result.total_chunks}")
    print(f"bad chunks: {result.bad_chunk_count}/{result.total_chunks}")

    if result.problems:
        print("\nPROBLEMS")
        print("-" * 80)
        for problem in result.problems:
            print(f"- {problem}")


def print_chunk_scores(result: LLMRetrieverMetricResult) -> None:
    print("\nPER-CHUNK LLM SCORES")
    print("-" * 80)

    for score in result.chunk_scores:
        print(f"[{score.rank}] {score.chunk_id} | page {score.page}")
        print(
            f"  label={score.label} "
            f"sem={score.semantic_relevance} "
            f"ev={score.evidence_value} "
            f"src={score.final_source_quality}"
        )
        print(f"  reason: {score.reason}")

        if score.problems:
            print(f"  problems: {score.problems}")


def print_summary_table(rows: list[dict]) -> None:
    print("\n\nUNSTRUCTURED RETRIEVER SUMMARY TABLE")
    print("=" * 150)

    print(
        f"{'question_id':<22}"
        f"{'k':>5}"
        f"{'final':>8}"
        f"{'sem':>8}"
        f"{'evid':>8}"
        f"{'src_q':>8}"
        f"{'eff':>8}"
        f"{'central':>9}"
        f"{'support':>9}"
        f"{'junk':>7}"
        f"{'bad':>7}"
    )
    print("-" * 150)

    for row in rows:
        print(
            f"{row['question_id']:<22}"
            f"{row['k']:>5}"
            f"{row['final_score']:>8.3f}"
            f"{row['average_semantic_relevance']:>8.3f}"
            f"{row['average_evidence_value']:>8.3f}"
            f"{row['average_source_quality']:>8.3f}"
            f"{row['context_efficiency']:>8.3f}"
            f"{row['central_count']:>9}"
            f"{row['supporting_count']:>9}"
            f"{row['junk_count']:>7}"
            f"{row['bad_chunk_count']:>7}"
        )

    print("=" * 150)

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


def run_matrix() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing chunks: {CHUNKS_PATH}. "
            f"Run: PYTHONPATH=src python -m book_rag.ingest.ingest_unstructured"
        )

    configure_dspy_for_metrics()
    judge = dspy.Predict(JudgeRetrievedChunk)

    rows: list[dict] = []

    for question in QUESTIONS:
        for k in K_VALUES:
            print("\n" + "=" * 120)
            print(f"unstructured_fast | {question.id} | k={k}")
            print(question.question)
            print("=" * 120)

            retriever = BookRetriever(
                chunks_path=CHUNKS_PATH,
                k=k,
            )
            chunks = retriever.search(question.question, k=k)

            result = calculate_llm_retriever_metrics(
                question=question.question,
                chunks=chunks,
                config=question.config,
                judge=judge,
            )

            print_llm_metrics(result)
            print_chunk_scores(result)

            rows.append(
                {
                    "question_id": question.id,
                    "k": k,
                    "final_score": result.final_score,
                    "average_semantic_relevance": result.average_semantic_relevance,
                    "average_evidence_value": result.average_evidence_value,
                    "average_source_quality": result.average_source_quality,
                    "context_efficiency": result.context_efficiency,
                    "central_count": result.central_count,
                    "supporting_count": result.supporting_count,
                    "junk_count": result.junk_count,
                    "bad_chunk_count": result.bad_chunk_count,
                }
            )

    print_summary_table(rows)


def main() -> None:
    run_matrix()


if __name__ == "__main__":
    main()