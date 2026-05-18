import argparse
from dataclasses import dataclass
from pathlib import Path

import dspy

from book_rag.answer_metrics_llm import (
    StructuredAnswerMetricResult,
    StructuredRAGAnswerJudge,
    calculate_structured_answer_metrics,
)
from book_rag.evaluation.artifacts import (
    StructuredEvalResult,
    save_answer_eval_artifacts,
)
from book_rag.programs import BookRAG
from book_rag.retriever.retriever import BookRetriever
from book_rag.retriever.retriever_metrics_llm import configure_dspy_for_metrics


@dataclass
class EvalQuestion:
    id: str
    question: str


CHUNKS_PATH = Path("data/processed/chunks_unstructured.jsonl")


QUESTIONS = [
    EvalQuestion(
        id="support_001",
        question="what does the book say about customer support agents?",
    ),
    EvalQuestion(
        id="cognitive_loop_001",
        question="what is the cognitive loop?",
    ),
    EvalQuestion(
        id="coding_agent_001",
        question="which parts are relevant to building a coding agent?",
    ),
]


K_VALUES = [5, 10, 20]


def print_answer(question_id: str, k: int, prediction) -> None:
    print("\n" + "=" * 120)
    print(f"unstructured_fast | {question_id} | k={k}")
    print("=" * 120)

    print("\nDIRECT ANSWER")
    print("-" * 80)
    print(prediction.direct_answer)

    print("\nWHAT THE BOOK SAYS")
    print("-" * 80)
    print(prediction.book_summary)

    print("\nKEY POINTS")
    print("-" * 80)
    if prediction.key_points:
        for point in prediction.key_points:
            print(f"- {point}")
    else:
        print("-")

    print("\nPRACTICAL USAGE")
    print("-" * 80)
    print(prediction.practical_usage)

    print("\nDESIGN TAKEAWAY")
    print("-" * 80)
    print(prediction.design_takeaway)

    print("\nCAVEATS")
    print("-" * 80)
    if prediction.caveats:
        for caveat in prediction.caveats:
            print(f"- {caveat}")
    else:
        print("-")

    print("\nFINAL ANSWER")
    print("-" * 80)
    print(prediction.answer)


def print_claim_scores(result: StructuredAnswerMetricResult) -> None:
    print("\nCLAIM-LEVEL SCORES")
    print("-" * 80)

    if not result.claim_scores:
        print("No claims extracted.")
        return

    for index, claim in enumerate(result.claim_scores, start=1):
        print(f"\n[{index}] {claim.claim}")
        print(f"  field: {claim.field_name}")
        print(f"  type: {claim.claim_type}")
        print(f"  support_status: {claim.support_status}")
        print(f"  support_score: {claim.support_score}")
        print(f"  example_error_type: {claim.example_error_type}")
        print(f"  proposed_sources: {claim.supporting_sources}")
        print(f"  matched_source_ids: {claim.matched_source_ids}")
        print(f"  has_valid_source: {claim.has_valid_source}")
        print(f"  source_alignment: {claim.source_alignment}")

        if claim.reason:
            print(f"  reason: {claim.reason}")

        if claim.problems:
            print(f"  problems: {claim.problems}")


def print_answer_metrics(result: StructuredAnswerMetricResult) -> None:
    print("\nSTRUCTURED ANSWER METRICS")
    print("-" * 80)
    print(f"final_score: {result.final_score}")
    print(f"directness: {result.directness.score}")
    print(f"book_grounding: {result.book_grounding.score}")
    print(f"key_point_coverage: {result.key_point_coverage.score}")
    print(f"practical_usefulness: {result.practical_usefulness.score}")
    print(f"design_takeaway_quality: {result.design_takeaway_quality.score}")
    print(f"caveat_quality: {result.caveat_quality.score}")
    print(f"final_answer_consistency: {result.final_answer_consistency.score}")
    print(f"citation_alignment: {result.citation_alignment.score}")
    print(f"example_discipline: {result.example_discipline.score}")
    print(f"clarity: {result.clarity.score}")

    print("\nREASONS")
    print("-" * 80)
    print(f"directness: {result.directness.reason}")
    print(f"book_grounding: {result.book_grounding.reason}")
    print(f"key_point_coverage: {result.key_point_coverage.reason}")
    print(f"practical_usefulness: {result.practical_usefulness.reason}")
    print(f"design_takeaway_quality: {result.design_takeaway_quality.reason}")
    print(f"caveat_quality: {result.caveat_quality.reason}")
    print(f"final_answer_consistency: {result.final_answer_consistency.reason}")
    print(f"citation_alignment: {result.citation_alignment.reason}")
    print(f"example_discipline: {result.example_discipline.reason}")
    print(f"clarity: {result.clarity.reason}")

    if result.missing_points:
        print("\nMISSING POINTS")
        print("-" * 80)
        for point in result.missing_points:
            print(f"- {point}")

    if result.problems:
        print("\nPROBLEMS")
        print("-" * 80)
        for problem in result.problems:
            print(f"- {problem}")

    print_claim_scores(result)

    print("\nRETRIEVED SOURCES")
    print("-" * 80)
    for source in result.retrieved_sources:
        print(f"- {source}")


def print_summary_table(rows: list[dict]) -> None:
    print("\n\nSTRUCTURED ANSWER EVAL SUMMARY TABLE")
    print("=" * 220)

    print(
        f"{'question_id':<22}"
        f"{'k':>5}"
        f"{'final':>8}"
        f"{'direct':>8}"
        f"{'ground':>8}"
        f"{'keypts':>8}"
        f"{'pract':>8}"
        f"{'design':>8}"
        f"{'caveat':>8}"
        f"{'consist':>9}"
        f"{'cit':>8}"
        f"{'ex':>7}"
        f"{'clar':>8}"
        f"{'claims':>8}"
        f"{'valid_src':>10}"
        f"{'missing':>9}"
    )
    print("-" * 220)

    for row in rows:
        print(
            f"{row['question_id']:<22}"
            f"{row['k']:>5}"
            f"{row['final_score']:>8.3f}"
            f"{row['directness']:>8.3f}"
            f"{row['book_grounding']:>8.3f}"
            f"{row['key_point_coverage']:>8.3f}"
            f"{row['practical_usefulness']:>8.3f}"
            f"{row['design_takeaway_quality']:>8.3f}"
            f"{row['caveat_quality']:>8.3f}"
            f"{row['final_answer_consistency']:>9.3f}"
            f"{row['citation_alignment']:>8.3f}"
            f"{row['example_discipline']:>7.3f}"
            f"{row['clarity']:>8.3f}"
            f"{row['claim_count']:>8}"
            f"{row['valid_source_count']:>10}"
            f"{row['missing_count']:>9}"
        )

    print("=" * 220)


def print_best_by_question(rows: list[dict]) -> None:
    print("\nBEST ANSWER CONFIG PER QUESTION")
    print("-" * 120)

    for question in QUESTIONS:
        question_rows = [
            row for row in rows if row["question_id"] == question.id
        ]

        best_row = max(question_rows, key=lambda row: row["final_score"])

        print(
            f"{question.id}: "
            f"k={best_row['k']} "
            f"(final={best_row['final_score']:.3f}, "
            f"ground={best_row['book_grounding']:.3f}, "
            f"practical={best_row['practical_usefulness']:.3f}, "
            f"design={best_row['design_takeaway_quality']:.3f}, "
            f"consistency={best_row['final_answer_consistency']:.3f}, "
            f"claims={best_row['claim_count']})"
        )


def print_average(rows: list[dict]) -> None:
    print("\nAVERAGE")
    print("-" * 120)

    avg_final = sum(row["final_score"] for row in rows) / len(rows)
    avg_direct = sum(row["directness"] for row in rows) / len(rows)
    avg_ground = sum(row["book_grounding"] for row in rows) / len(rows)
    avg_keypts = sum(row["key_point_coverage"] for row in rows) / len(rows)
    avg_practical = sum(row["practical_usefulness"] for row in rows) / len(rows)
    avg_design = sum(row["design_takeaway_quality"] for row in rows) / len(rows)
    avg_caveat = sum(row["caveat_quality"] for row in rows) / len(rows)
    avg_consistency = sum(row["final_answer_consistency"] for row in rows) / len(rows)
    avg_citation = sum(row["citation_alignment"] for row in rows) / len(rows)
    avg_example = sum(row["example_discipline"] for row in rows) / len(rows)
    avg_clarity = sum(row["clarity"] for row in rows) / len(rows)
    avg_claims = sum(row["claim_count"] for row in rows) / len(rows)
    avg_valid_sources = sum(row["valid_source_count"] for row in rows) / len(rows)
    avg_missing = sum(row["missing_count"] for row in rows) / len(rows)

    print(
        f"avg_final={avg_final:.3f}, "
        f"avg_direct={avg_direct:.3f}, "
        f"avg_ground={avg_ground:.3f}, "
        f"avg_keypts={avg_keypts:.3f}, "
        f"avg_practical={avg_practical:.3f}, "
        f"avg_design={avg_design:.3f}, "
        f"avg_caveat={avg_caveat:.3f}, "
        f"avg_consistency={avg_consistency:.3f}, "
        f"avg_citation={avg_citation:.3f}, "
        f"avg_example={avg_example:.3f}, "
        f"avg_clarity={avg_clarity:.3f}, "
        f"avg_claims={avg_claims:.2f}, "
        f"avg_valid_sources={avg_valid_sources:.2f}, "
        f"avg_missing={avg_missing:.2f}"
    )


def run_answer_eval(run_name: str | None = None) -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing chunks: {CHUNKS_PATH}. "
            f"Run: PYTHONPATH=src python -m book_rag.ingest.ingest_unstructured"
        )

    configure_dspy_for_metrics()

    judge = dspy.Predict(StructuredRAGAnswerJudge)

    rows: list[dict] = []
    all_results: list[StructuredEvalResult] = []

    for question in QUESTIONS:
        for k in K_VALUES:
            retriever = BookRetriever(
                chunks_path=CHUNKS_PATH,
                k=k,
            )

            rag = BookRAG(
                retriever=retriever,
                k=k,
            )

            prediction = rag(question=question.question)

            answer_result = calculate_structured_answer_metrics(
                question=question.question,
                direct_answer=prediction.direct_answer,
                book_summary=prediction.book_summary,
                key_points=prediction.key_points,
                practical_usage=prediction.practical_usage,
                design_takeaway=prediction.design_takeaway,
                caveats=prediction.caveats,
                final_answer=prediction.answer,
                chunks=prediction.chunks,
                judge=judge,
            )

            print_answer(
                question_id=question.id,
                k=k,
                prediction=prediction,
            )

            print_answer_metrics(answer_result)

            all_results.append(
                StructuredEvalResult(
                    question_id=question.id,
                    k=k,
                    metrics=answer_result,
                )
            )

            valid_source_count = sum(
                1 for claim in answer_result.claim_scores if claim.has_valid_source
            )

            rows.append(
                {
                    "question_id": question.id,
                    "k": k,
                    "final_score": answer_result.final_score,
                    "directness": answer_result.directness.score,
                    "book_grounding": answer_result.book_grounding.score,
                    "key_point_coverage": answer_result.key_point_coverage.score,
                    "practical_usefulness": answer_result.practical_usefulness.score,
                    "design_takeaway_quality": answer_result.design_takeaway_quality.score,
                    "caveat_quality": answer_result.caveat_quality.score,
                    "final_answer_consistency": answer_result.final_answer_consistency.score,
                    "citation_alignment": answer_result.citation_alignment.score,
                    "example_discipline": answer_result.example_discipline.score,
                    "clarity": answer_result.clarity.score,
                    "claim_count": len(answer_result.claim_scores),
                    "valid_source_count": valid_source_count,
                    "missing_count": len(answer_result.missing_points),
                }
            )

    print_summary_table(rows)
    print_best_by_question(rows)
    print_average(rows)

    run_dir = save_answer_eval_artifacts(
        results=all_results,
        run_name=run_name,
    )
    print(f"\nSaved evaluation artifacts to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for the evaluation artifact directory.",
    )
    args = parser.parse_args()

    run_answer_eval(run_name=args.run_name)


if __name__ == "__main__":
    main()