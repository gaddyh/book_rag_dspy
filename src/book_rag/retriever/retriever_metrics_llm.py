import json
import os
from typing import Literal

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from book_rag.retriever.retriever_metrics import (
    RetrieverMetricConfig,
    score_bad_source,
    score_chunk_relevance,
)


ChunkLabel = Literal[
    "central",
    "supporting",
    "background",
    "irrelevant",
    "junk",
]


class LLMChunkJudgeScore(BaseModel):
    chunk_id: str
    page: int | None
    rank: int

    semantic_relevance: float = Field(ge=0.0, le=1.0)
    evidence_value: float = Field(ge=0.0, le=1.0)
    llm_source_quality: float = Field(ge=0.0, le=1.0)
    deterministic_source_quality: float = Field(ge=0.0, le=1.0)
    final_source_quality: float = Field(ge=0.0, le=1.0)

    relevance_hits: list[str] = Field(default_factory=list)
    bad_source_reasons: list[str] = Field(default_factory=list)

    label: ChunkLabel
    reason: str
    problems: list[str] = Field(default_factory=list)

    text_preview: str


class LLMRetrieverMetricResult(BaseModel):
    question: str
    k: int

    average_semantic_relevance: float
    average_evidence_value: float
    average_source_quality: float

    central_count: int
    supporting_count: int
    background_count: int
    irrelevant_count: int
    junk_count: int

    context_efficiency: float
    useful_chunk_count: int
    bad_chunk_count: int
    total_chunks: int

    final_score: float

    chunk_scores: list[LLMChunkJudgeScore]
    problems: list[str] = Field(default_factory=list)


class JudgeRetrievedChunk(dspy.Signature):
    """
    Judge whether a retrieved chunk is useful evidence for answering a question.

    Score strictly.

    semantic_relevance:
    - 1.0 = directly answers the question
    - 0.7 = strongly related and useful
    - 0.4 = loosely related/background
    - 0.0 = irrelevant

    evidence_value:
    - 1.0 = central evidence needed for the answer
    - 0.7 = useful supporting evidence
    - 0.3 = background context only
    - 0.0 = no evidence value

    source_quality:
    - 1.0 = clean body content
    - 0.6 = usable but partial/fragmented
    - 0.2 = TOC/front matter/footer/very short fragment
    - 0.0 = junk

    label:
    central, supporting, background, irrelevant, or junk.
    """

    question: str = dspy.InputField()
    chunk_text: str = dspy.InputField()
    chunk_metadata: str = dspy.InputField()

    semantic_relevance: float = dspy.OutputField(
        desc="Float from 0.0 to 1.0. How semantically relevant is this chunk to the question?"
    )
    evidence_value: float = dspy.OutputField(
        desc="Float from 0.0 to 1.0. How valuable is this chunk as evidence for answering?"
    )
    source_quality: float = dspy.OutputField(
        desc="Float from 0.0 to 1.0. Is this clean body content or noisy source text?"
    )
    label: ChunkLabel = dspy.OutputField(
        desc="One of: central, supporting, background, irrelevant, junk."
    )
    reason: str = dspy.OutputField(desc="Short reason for the judgment.")
    problems: list[str] = dspy.OutputField(
        desc="Specific problems found, if any. Empty list if none."
    )


def configure_dspy_for_metrics(
    model: str = "openai/gpt-4o-mini",
) -> None:
    """
    Configure DSPy for LLM-as-judge metrics.

    Safe to call multiple times.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or export it in your shell."
        )

    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)


def clamp_score(value: object, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default

    return max(0.0, min(1.0, score))


def normalize_label(label: object) -> ChunkLabel:
    if not isinstance(label, str):
        return "background"

    label = label.strip().lower()

    allowed = {
        "central",
        "supporting",
        "background",
        "irrelevant",
        "junk",
    }

    if label in allowed:
        return label  # type: ignore[return-value]

    return "background"


def normalize_problems(problems: object) -> list[str]:
    if problems is None:
        return []

    if isinstance(problems, list):
        return [str(problem) for problem in problems if str(problem).strip()]

    if isinstance(problems, str):
        stripped = problems.strip()
        if not stripped:
            return []

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(problem) for problem in parsed]
        except json.JSONDecodeError:
            pass

        return [stripped]

    return [str(problems)]


def deterministic_source_quality(
    chunk: dict,
    config: RetrieverMetricConfig,
) -> tuple[float, list[str]]:
    is_bad, reasons = score_bad_source(
        chunk=chunk,
        bad_page_max=config.bad_page_max,
        min_chunk_chars=config.min_chunk_chars,
    )

    if not is_bad:
        return 1.0, []

    # Simple severity rules.
    # TOC/front matter are bad, but sometimes still useful as weak navigation hints.
    if "too_short" in " ".join(reasons):
        return 0.4, reasons

    if "table_of_contents" in reasons:
        return 0.2, reasons

    if any(reason.startswith("front_matter_page") for reason in reasons):
        return 0.3, reasons

    return 0.2, reasons


def score_one_chunk_with_llm(
    judge: dspy.Predict,
    question: str,
    chunk: dict,
    rank: int,
    config: RetrieverMetricConfig,
) -> LLMChunkJudgeScore:
    text = chunk.get("text", "")
    page = chunk.get("page_start")
    chunk_id = chunk.get("chunk_id", "unknown")

    _, relevance_hits = score_chunk_relevance(
        text=text,
        keywords=config.relevance_keywords,
    )

    deterministic_quality, bad_source_reasons = deterministic_source_quality(
        chunk=chunk,
        config=config,
    )

    metadata = {
        "chunk_id": chunk_id,
        "page": page,
        "rank": rank,
        "has_images": chunk.get("has_images"),
        "image_count": chunk.get("image_count"),
        "deterministic_bad_source_reasons": bad_source_reasons,
        "keyword_hits": relevance_hits,
    }

    prediction = judge(
        question=question,
        chunk_text=text,
        chunk_metadata=json.dumps(metadata, ensure_ascii=False),
    )

    semantic_relevance = clamp_score(prediction.semantic_relevance)
    evidence_value = clamp_score(prediction.evidence_value)
    llm_source_quality = clamp_score(prediction.source_quality)

    # Deterministic quality is a guardrail. If rules say source is bad,
    # don't let the LLM fully forgive it.
    final_source_quality = min(deterministic_quality, llm_source_quality)

    label = normalize_label(prediction.label)
    problems = normalize_problems(prediction.problems)

    if bad_source_reasons:
        problems.extend([f"deterministic:{reason}" for reason in bad_source_reasons])

    return LLMChunkJudgeScore(
        chunk_id=chunk_id,
        page=page,
        rank=rank,
        semantic_relevance=round(semantic_relevance, 3),
        evidence_value=round(evidence_value, 3),
        llm_source_quality=round(llm_source_quality, 3),
        deterministic_source_quality=round(deterministic_quality, 3),
        final_source_quality=round(final_source_quality, 3),
        relevance_hits=relevance_hits,
        bad_source_reasons=bad_source_reasons,
        label=label,
        reason=str(prediction.reason),
        problems=problems,
        text_preview=text[:250],
    )


def calculate_llm_retriever_metrics(
    question: str,
    chunks: list[dict],
    config: RetrieverMetricConfig,
    judge: dspy.Predict | None = None,
) -> LLMRetrieverMetricResult:
    """
    LLM-as-judge retriever evaluation.

    Scores each retrieved chunk, then aggregates run-level metrics.
    """
    if judge is None:
        judge = dspy.Predict(JudgeRetrievedChunk)

    chunk_scores: list[LLMChunkJudgeScore] = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_scores.append(
            score_one_chunk_with_llm(
                judge=judge,
                question=question,
                chunk=chunk,
                rank=index,
                config=config,
            )
        )

    total = len(chunk_scores)

    if total == 0:
        return LLMRetrieverMetricResult(
            question=question,
            k=0,
            average_semantic_relevance=0.0,
            average_evidence_value=0.0,
            average_source_quality=0.0,
            central_count=0,
            supporting_count=0,
            background_count=0,
            irrelevant_count=0,
            junk_count=0,
            context_efficiency=0.0,
            useful_chunk_count=0,
            bad_chunk_count=0,
            total_chunks=0,
            final_score=0.0,
            chunk_scores=[],
            problems=["No chunks retrieved."],
        )

    average_semantic_relevance = sum(
        score.semantic_relevance for score in chunk_scores
    ) / total

    average_evidence_value = sum(
        score.evidence_value for score in chunk_scores
    ) / total

    average_source_quality = sum(
        score.final_source_quality for score in chunk_scores
    ) / total

    central_count = sum(1 for score in chunk_scores if score.label == "central")
    supporting_count = sum(1 for score in chunk_scores if score.label == "supporting")
    background_count = sum(1 for score in chunk_scores if score.label == "background")
    irrelevant_count = sum(1 for score in chunk_scores if score.label == "irrelevant")
    junk_count = sum(1 for score in chunk_scores if score.label == "junk")

    useful_chunk_count = central_count + supporting_count

    bad_chunk_count = sum(
        1
        for score in chunk_scores
        if score.final_source_quality < 0.5 or score.label == "junk"
    )

    context_efficiency = useful_chunk_count / total

    # Reward central/supporting evidence and source cleanliness.
    # Penalize bloated retrieval indirectly through context_efficiency.
    final_score = (
        0.35 * average_evidence_value
        + 0.30 * average_semantic_relevance
        + 0.20 * average_source_quality
        + 0.15 * context_efficiency
    )

    problems: list[str] = []

    if central_count == 0:
        problems.append("No central evidence chunks retrieved.")

    if useful_chunk_count == 0:
        problems.append("No useful evidence chunks retrieved.")

    if bad_chunk_count > 0:
        bad_ids = [
            score.chunk_id
            for score in chunk_scores
            if score.final_source_quality < 0.5 or score.label == "junk"
        ]
        problems.append(f"Bad/junk chunks retrieved: {bad_ids}")

    weak_ids = [
        score.chunk_id
        for score in chunk_scores
        if score.semantic_relevance < 0.5
    ]

    if weak_ids:
        problems.append(f"Low semantic relevance chunks: {weak_ids}")

    return LLMRetrieverMetricResult(
        question=question,
        k=total,
        average_semantic_relevance=round(average_semantic_relevance, 3),
        average_evidence_value=round(average_evidence_value, 3),
        average_source_quality=round(average_source_quality, 3),
        central_count=central_count,
        supporting_count=supporting_count,
        background_count=background_count,
        irrelevant_count=irrelevant_count,
        junk_count=junk_count,
        context_efficiency=round(context_efficiency, 3),
        useful_chunk_count=useful_chunk_count,
        bad_chunk_count=bad_chunk_count,
        total_chunks=total,
        final_score=round(final_score, 3),
        chunk_scores=chunk_scores,
        problems=problems,
    )