from dataclasses import dataclass
from pathlib import Path

import dspy

from book_rag.retriever import BookRetriever
from book_rag.retriever_metrics import RetrieverMetricConfig
from book_rag.retriever_metrics_llm import (
    LLMRetrieverMetricResult,
    JudgeRetrievedChunk,
    calculate_llm_retriever_metrics,
    configure_dspy_for_metrics,
)


@dataclass
class IngestionCandidate:
    id: str
    chunks_path: Path


@dataclass
class EvalQuestion:
    id: str
    question: str
    config: RetrieverMetricConfig


INGESTION_CANDIDATES = [
    IngestionCandidate(
        id="pymupdf",
        chunks_path=Path("data/processed/chunks.jsonl"),
    ),
    IngestionCandidate(
        id="unstructured_fast",
        chunks_path=Path("data/processed/chunks_unstructured.jsonl"),
    ),
]


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


def print_summary_table(rows: list[dict]) -> None:
    print("\n\nINGESTION COMPARISON TABLE")
    print("=" * 170)

    print(
        f"{'ingestion':<20}"
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
    print("-" * 170)

    for row in rows:
        print(
            f"{row['ingestion']:<20}"
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

    print("=" * 170)


def print_best_by_question(rows: list[dict]) -> None:
    print("\nBEST INGESTION/K PER QUESTION")
    print("-" * 100)

    for question in QUESTIONS:
        question_rows = [
            row for row in rows if row["question_id"] == question.id
        ]

        best_row = max(question_rows, key=lambda row: row["final_score"])

        print(
            f"{question.id}: "
            f"{best_row['ingestion']} k={best_row['k']} "
            f"(final={best_row['final_score']:.3f}, "
            f"sem={best_row['average_semantic_relevance']:.3f}, "
            f"eff={best_row['context_efficiency']:.3f})"
        )


def print_average_by_ingestion(rows: list[dict]) -> None:
    print("\nAVERAGE BY INGESTION")
    print("-" * 100)

    for candidate in INGESTION_CANDIDATES:
        candidate_rows = [
            row for row in rows if row["ingestion"] == candidate.id
        ]

        if not candidate_rows:
            continue

        avg_final = sum(row["final_score"] for row in candidate_rows) / len(candidate_rows)
        avg_sem = sum(row["average_semantic_relevance"] for row in candidate_rows) / len(candidate_rows)
        avg_eff = sum(row["context_efficiency"] for row in candidate_rows) / len(candidate_rows)
        avg_bad = sum(row["bad_chunk_count"] for row in candidate_rows) / len(candidate_rows)

        print(
            f"{candidate.id}: "
            f"avg_final={avg_final:.3f}, "
            f"avg_sem={avg_sem:.3f}, "
            f"avg_eff={avg_eff:.3f}, "
            f"avg_bad_chunks={avg_bad:.2f}"
        )


def run_comparison() -> None:
    configure_dspy_for_metrics()
    judge = dspy.Predict(JudgeRetrievedChunk)

    rows: list[dict] = []

    for candidate in INGESTION_CANDIDATES:
        if not candidate.chunks_path.exists():
            raise FileNotFoundError(
                f"Missing chunks for {candidate.id}: {candidate.chunks_path}. "
                f"Run the ingestion script first."
            )

        for question in QUESTIONS:
            for k in K_VALUES:
                print("\n" + "=" * 120)
                print(f"{candidate.id} | {question.id} | k={k}")
                print(question.question)
                print("=" * 120)

                retriever = BookRetriever(
                    chunks_path=candidate.chunks_path,
                    k=k,
                )

                chunks = retriever.search(question.question)

                result: LLMRetrieverMetricResult = calculate_llm_retriever_metrics(
                    question=question.question,
                    chunks=chunks,
                    config=question.config,
                    judge=judge,
                )

                print(
                    f"final={result.final_score} "
                    f"sem={result.average_semantic_relevance} "
                    f"evid={result.average_evidence_value} "
                    f"src_q={result.average_source_quality} "
                    f"eff={result.context_efficiency} "
                    f"central={result.central_count} "
                    f"support={result.supporting_count} "
                    f"junk={result.junk_count} "
                    f"bad={result.bad_chunk_count}"
                )

                rows.append(
                    {
                        "ingestion": candidate.id,
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
    print_best_by_question(rows)
    print_average_by_ingestion(rows)


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()