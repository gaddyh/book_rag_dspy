from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from book_rag.answer_metrics_llm import StructuredAnswerMetricResult


REPORTS_DIR = Path("reports") / "answer_eval"


@dataclass(frozen=True)
class StructuredEvalResult:
    question_id: str
    k: int
    metrics: StructuredAnswerMetricResult


def _to_jsonable(value: Any) -> Any:
    """
    Convert nested evaluation objects into JSON-serializable structures.

    Handles:
    - Pydantic BaseModel via model_dump()
    - dataclasses via asdict()
    - lists
    - dicts
    - primitive values
    """
    if isinstance(value, BaseModel):
        return value.model_dump()

    if is_dataclass(value):
        return {
            key: _to_jsonable(inner_value)
            for key, inner_value in asdict(value).items()
        }

    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(inner_value)
            for key, inner_value in value.items()
        }

    return value


def create_run_dir(run_name: str | None = None) -> Path:
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPORTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(results: list[StructuredEvalResult], path: Path) -> None:
    payload = [_to_jsonable(result) for result in results]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _score(metric: Any) -> float:
    """
    StructuredAnswerMetricResult.final_score is a float.
    The other metric fields are AnswerMetricScore objects with a .score field.
    """
    if isinstance(metric, int | float):
        return float(metric)
    score = getattr(metric, "score", None)
    if score is None:
        return 0.0
    return float(score)


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    return len(value)


def build_summary_rows(results: list[StructuredEvalResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for result in results:
        metrics = result.metrics

        claim_scores = getattr(metrics, "claim_scores", []) or []
        problems = getattr(metrics, "problems", []) or []

        valid_sources = sum(
            1
            for claim in claim_scores
            if getattr(claim, "has_valid_source", False)
        )

        rows.append(
            {
                "question_id": result.question_id,
                "k": result.k,
                "final_score": float(metrics.final_score),
                "directness": _score(metrics.directness),
                "book_grounding": _score(metrics.book_grounding),
                "key_point_coverage": _score(metrics.key_point_coverage),
                "practical_usefulness": _score(metrics.practical_usefulness),
                "design_takeaway_quality": _score(metrics.design_takeaway_quality),
                "caveat_quality": _score(metrics.caveat_quality),
                "final_answer_consistency": _score(metrics.final_answer_consistency),
                "citation_alignment": _score(metrics.citation_alignment),
                "example_discipline": _score(metrics.example_discipline),
                "clarity": _score(metrics.clarity),
                "claims": _safe_len(claim_scores),
                "valid_sources": valid_sources,
                "missing": _safe_len(problems),
            }
        )

    return rows


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def save_markdown_report(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text(
            "# Structured Answer Evaluation Report\n\nNo results.\n",
            encoding="utf-8",
        )
        return

    lines = [
        "# Structured Answer Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Runs evaluated: {len(rows)}",
        f"- Average final score: {_avg(rows, 'final_score'):.3f}",
        f"- Average directness: {_avg(rows, 'directness'):.3f}",
        f"- Average book grounding: {_avg(rows, 'book_grounding'):.3f}",
        f"- Average key point coverage: {_avg(rows, 'key_point_coverage'):.3f}",
        f"- Average practical usefulness: {_avg(rows, 'practical_usefulness'):.3f}",
        f"- Average design takeaway quality: {_avg(rows, 'design_takeaway_quality'):.3f}",
        f"- Average caveat quality: {_avg(rows, 'caveat_quality'):.3f}",
        f"- Average final answer consistency: {_avg(rows, 'final_answer_consistency'):.3f}",
        f"- Average citation alignment: {_avg(rows, 'citation_alignment'):.3f}",
        f"- Average example discipline: {_avg(rows, 'example_discipline'):.3f}",
        f"- Average clarity: {_avg(rows, 'clarity'):.3f}",
        "",
        "## Results",
        "",
        "| Question ID | k | Final | Direct | Ground | KeyPts | Practical | Design | Caveat | Consistency | Citations | Claims | Valid Sources | Missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['question_id']} | "
            f"{row['k']} | "
            f"{row['final_score']:.3f} | "
            f"{row['directness']:.3f} | "
            f"{row['book_grounding']:.3f} | "
            f"{row['key_point_coverage']:.3f} | "
            f"{row['practical_usefulness']:.3f} | "
            f"{row['design_takeaway_quality']:.3f} | "
            f"{row['caveat_quality']:.3f} | "
            f"{row['final_answer_consistency']:.3f} | "
            f"{row['citation_alignment']:.3f} | "
            f"{row['claims']} | "
            f"{row['valid_sources']} | "
            f"{row['missing']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_answer_eval_artifacts(
    results: list[StructuredEvalResult],
    run_name: str | None = None,
) -> Path:
    run_dir = create_run_dir(run_name)
    rows = build_summary_rows(results)

    save_json(results, run_dir / "results.json")
    save_summary_csv(rows, run_dir / "summary.csv")
    save_markdown_report(rows, run_dir / "report.md")

    return run_dir
