import json
from typing import Any, Literal

import dspy
from pydantic import BaseModel, Field


ClaimSupportStatus = Literal["supported", "partial", "unsupported"]

ClaimType = Literal[
    "definition",
    "general_claim",
    "specific_example",
    "interpretation",
    "recommendation",
    "unsupported_or_unclear",
]

ExampleErrorType = Literal[
    "none",
    "example_as_universal_rule",
    "case_study_as_recommendation",
    "product_result_as_general_result",
]


class AnswerClaimScore(BaseModel):
    claim: str
    field_name: str = Field(
        description=(
            "Which answer field this claim came from: direct_answer, book_summary, "
            "key_points, practical_usage, design_takeaway, caveats, or final_answer."
        )
    )
    claim_type: ClaimType = "unsupported_or_unclear"

    support_status: ClaimSupportStatus = "unsupported"
    support_score: float = Field(ge=0.0, le=1.0)

    supporting_sources: list[str] = Field(default_factory=list)
    matched_source_ids: list[str] = Field(default_factory=list)
    has_valid_source: bool = False
    source_alignment: float = Field(ge=0.0, le=1.0)

    example_error_type: ExampleErrorType = "none"

    reason: str = ""
    problems: list[str] = Field(default_factory=list)


class AnswerMetricScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    problems: list[str] = Field(default_factory=list)


class StructuredAnswerMetricResult(BaseModel):
    question: str

    direct_answer: str
    book_summary: str
    key_points: list[str]
    practical_usage: str
    design_takeaway: str
    caveats: list[str]
    final_answer: str

    retrieved_sources: list[str]

    directness: AnswerMetricScore
    book_grounding: AnswerMetricScore
    key_point_coverage: AnswerMetricScore
    practical_usefulness: AnswerMetricScore
    design_takeaway_quality: AnswerMetricScore
    caveat_quality: AnswerMetricScore
    final_answer_consistency: AnswerMetricScore
    citation_alignment: AnswerMetricScore
    example_discipline: AnswerMetricScore
    clarity: AnswerMetricScore

    claim_scores: list[AnswerClaimScore] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)

    final_score: float
    problems: list[str] = Field(default_factory=list)


class StructuredRAGAnswerJudge(dspy.Signature):
    """
    Judge a structured RAG answer.

    Use only the retrieved context as evidence.

    The answer has multiple fields:
    - direct_answer: should answer immediately and briefly.
    - book_summary: must stay close to what the retrieved context says.
    - key_points: should capture the important grounded points.
    - practical_usage: may interpret the book for practical AI engineering use, but must not pretend
      interpretation is directly stated by the book.
    - design_takeaway: should be a useful engineering/design lesson.
    - caveats: should be meaningful, not generic filler.
    - final_answer: should combine the fields without losing important components.

    Be strict:
    - Do not give 1.0 unless the field is genuinely excellent.
    - Penalize generic advice.
    - Penalize final_answer if it omits key points from direct_answer/key_points.
    - Penalize practical_usage if it is vague and not actionable.
    - Penalize caveats if they are generic filler.
    - Penalize book_summary if it includes practical interpretation as if it were book content.

    Source rules:
    - supporting_sources MUST contain exact chunk IDs copied from retrieved_sources.
    - Do not put quotes in supporting_sources.
    - Do not invent source IDs.
    - If no retrieved chunk supports the claim, use an empty supporting_sources list.

    Return JSON strings for claim_scores_json, missing_points_json, and overall_problems_json.
    Do not wrap JSON in markdown.
    """

    question: str = dspy.InputField()
    direct_answer: str = dspy.InputField()
    book_summary: str = dspy.InputField()
    key_points: str = dspy.InputField(desc="JSON array of key point strings.")
    practical_usage: str = dspy.InputField()
    design_takeaway: str = dspy.InputField()
    caveats: str = dspy.InputField(desc="JSON array of caveat strings.")
    final_answer: str = dspy.InputField()

    retrieved_context: str = dspy.InputField()
    retrieved_sources: str = dspy.InputField(
        desc=(
            "JSON array of retrieved source strings. Each source starts with the exact chunk ID. "
            "Use these exact chunk IDs in supporting_sources."
        )
    )

    claim_scores_json: str = dspy.OutputField(
        desc=(
            "JSON array. Each item must have: claim, field_name, claim_type, "
            "support_status, support_score, supporting_sources, example_error_type, "
            "reason, problems. supporting_sources must be exact retrieved chunk IDs only."
        )
    )

    missing_points_json: str = dspy.OutputField(
        desc=(
            "JSON array of important points from the retrieved context that the answer "
            "should have included but missed. Empty array if none."
        )
    )

    directness_score: float = dspy.OutputField(
        desc="0.0 to 1.0. Does direct_answer answer the question immediately, correctly, and without fluff?"
    )
    directness_reason: str = dspy.OutputField()

    key_point_coverage_score: float = dspy.OutputField(
        desc=(
            "0.0 to 1.0. Do key_points cover the important grounded points from the retrieved context?"
        )
    )
    key_point_coverage_reason: str = dspy.OutputField()

    practical_usefulness_score: float = dspy.OutputField(
        desc=(
            "0.0 to 1.0. Is practical_usage concrete, actionable, and useful for an AI engineer "
            "building real RAG/agent systems? Penalize generic advice."
        )
    )
    practical_usefulness_reason: str = dspy.OutputField()

    design_takeaway_quality_score: float = dspy.OutputField(
        desc=(
            "0.0 to 1.0. Is design_takeaway a sharp engineering lesson, not a vague slogan?"
        )
    )
    design_takeaway_quality_reason: str = dspy.OutputField()

    caveat_quality_score: float = dspy.OutputField(
        desc=(
            "0.0 to 1.0. Are caveats meaningful and evidence-aware? Penalize generic caveats like "
            "'implementation may vary' unless they are tied to retrieved context or system limits."
        )
    )
    caveat_quality_reason: str = dspy.OutputField()

    final_answer_consistency_score: float = dspy.OutputField(
        desc=(
            "0.0 to 1.0. Does final_answer faithfully combine the structured fields without "
            "dropping important components or adding unsupported claims?"
        )
    )
    final_answer_consistency_reason: str = dspy.OutputField()

    clarity_score: float = dspy.OutputField(
        desc="0.0 to 1.0. Is the overall structured answer clear and readable?"
    )
    clarity_reason: str = dspy.OutputField()

    overall_problems_json: str = dspy.OutputField(
        desc="JSON array of the most important answer-level problems."
    )


def clamp_score(value: object, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default

    return max(0.0, min(1.0, score))


def safe_json_loads(value: object, fallback: Any) -> Any:
    if value is None:
        return fallback

    if isinstance(value, (list, dict)):
        return value

    if not isinstance(value, str):
        return fallback

    text = value.strip()

    if not text:
        return fallback

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def normalize_list(value: object) -> list[str]:
    parsed = safe_json_loads(value, fallback=None)

    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]

    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def normalize_claim_type(value: object) -> ClaimType:
    allowed: set[str] = {
        "definition",
        "general_claim",
        "specific_example",
        "interpretation",
        "recommendation",
        "unsupported_or_unclear",
    }

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in allowed:
            return cleaned  # type: ignore[return-value]

    return "unsupported_or_unclear"


def normalize_support_status(value: object) -> ClaimSupportStatus:
    allowed: set[str] = {"supported", "partial", "unsupported"}

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in allowed:
            return cleaned  # type: ignore[return-value]

    return "unsupported"


def normalize_example_error_type(value: object) -> ExampleErrorType:
    allowed: set[str] = {
        "none",
        "example_as_universal_rule",
        "case_study_as_recommendation",
        "product_result_as_general_result",
    }

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in allowed:
            return cleaned  # type: ignore[return-value]

    return "none"


def status_to_score(status: ClaimSupportStatus, model_score: float) -> float:
    model_score = clamp_score(model_score)

    if status == "supported":
        return max(0.75, model_score)

    if status == "partial":
        return min(0.7, max(0.35, model_score))

    return min(0.25, model_score)


def stringify_retrieved_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        has_images = chunk.get("has_images", False)
        image_count = chunk.get("image_count", 0)
        text = chunk.get("text", "")

        if page_start is not None and page_end is not None and page_end != page_start:
            page_label = f"pages {page_start}-{page_end}"
        elif page_start is not None:
            page_label = f"page {page_start}"
        else:
            page_label = "page unknown"

        parts.append(
            f"[{chunk_id} | {page_label} | images={has_images} ({image_count})]\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(parts)


def stringify_retrieved_sources(chunks: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")

        if page_start is not None and page_end is not None and page_end != page_start:
            page_label = f"pages {page_start}-{page_end}"
        elif page_start is not None:
            page_label = f"page {page_start}"
        else:
            page_label = "page unknown"

        sources.append(f"{chunk_id} | {page_label}")

    return sources


def get_retrieved_chunk_ids(chunks: list[dict[str, Any]]) -> set[str]:
    return {
        str(chunk.get("chunk_id"))
        for chunk in chunks
        if chunk.get("chunk_id")
    }


def extract_chunk_id_from_source_ref(source_ref: str) -> str:
    return source_ref.split("|", 1)[0].strip()


def find_matching_source_ids(
    raw_sources: list[str],
    retrieved_chunk_ids: set[str],
) -> list[str]:
    matches: list[str] = []

    for raw_source in raw_sources:
        raw = str(raw_source).strip()
        if not raw:
            continue

        possible_id = extract_chunk_id_from_source_ref(raw)

        if possible_id in retrieved_chunk_ids:
            matches.append(possible_id)
            continue

        for chunk_id in retrieved_chunk_ids:
            if chunk_id in raw:
                matches.append(chunk_id)

    deduped: list[str] = []
    seen: set[str] = set()

    for match in matches:
        if match not in seen:
            deduped.append(match)
            seen.add(match)

    return deduped


def parse_claim_scores(
    raw_claims: object,
    retrieved_chunk_ids: set[str],
) -> tuple[list[AnswerClaimScore], list[str]]:
    parsed = safe_json_loads(raw_claims, fallback=[])

    if not isinstance(parsed, list):
        return [], ["Could not parse claim_scores_json as a JSON array."]

    claim_scores: list[AnswerClaimScore] = []
    parse_problems: list[str] = []

    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            parse_problems.append(f"Claim item {index} was not an object.")
            continue

        claim = str(item.get("claim", "")).strip()
        if not claim:
            parse_problems.append(f"Claim item {index} had empty claim text.")
            continue

        support_status = normalize_support_status(item.get("support_status"))
        support_score = status_to_score(
            support_status,
            clamp_score(item.get("support_score")),
        )

        raw_supporting_sources = item.get("supporting_sources", [])
        if not isinstance(raw_supporting_sources, list):
            raw_supporting_sources = [str(raw_supporting_sources)]

        supporting_sources = [
            str(source).strip()
            for source in raw_supporting_sources
            if str(source).strip()
        ]

        matched_source_ids = find_matching_source_ids(
            raw_sources=supporting_sources,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

        problems = item.get("problems", [])
        if not isinstance(problems, list):
            problems = [str(problems)] if problems else []

        if support_status in {"supported", "partial"} and not matched_source_ids:
            problems.append(
                "Claim marked supported/partial but no valid retrieved chunk ID was cited."
            )

        source_alignment = 1.0 if support_status in {"supported", "partial"} and matched_source_ids else 0.0

        claim_scores.append(
            AnswerClaimScore(
                claim=claim,
                field_name=str(item.get("field_name", "unknown")).strip(),
                claim_type=normalize_claim_type(item.get("claim_type")),
                support_status=support_status,
                support_score=round(support_score, 3),
                supporting_sources=supporting_sources,
                matched_source_ids=matched_source_ids,
                has_valid_source=bool(matched_source_ids),
                source_alignment=round(source_alignment, 3),
                example_error_type=normalize_example_error_type(
                    item.get("example_error_type")
                ),
                reason=str(item.get("reason", "")).strip(),
                problems=[str(problem) for problem in problems if str(problem).strip()],
            )
        )

    if not claim_scores:
        parse_problems.append("No valid answer claims were extracted.")

    return claim_scores, parse_problems


def metric_score(
    score: float,
    reason: str,
    problems: list[str] | None = None,
) -> AnswerMetricScore:
    return AnswerMetricScore(
        score=round(clamp_score(score), 3),
        reason=reason,
        problems=problems or [],
    )


def build_book_grounding_metric(
    claim_scores: list[AnswerClaimScore],
) -> AnswerMetricScore:
    """
    Grounding for book-grounded fields.

    Practical interpretation is allowed to be more interpretive, so this metric
    focuses mostly on direct_answer, book_summary, key_points, caveats, final_answer.
    """
    grounded_fields = {
        "direct_answer",
        "book_summary",
        "key_points",
        "caveats",
        "final_answer",
    }

    relevant_claims = [
        claim for claim in claim_scores if claim.field_name in grounded_fields
    ]

    if not relevant_claims:
        relevant_claims = claim_scores

    if not relevant_claims:
        return metric_score(
            0.0,
            "No claims were extracted, so grounding cannot be verified.",
            ["No valid answer claims were extracted."],
        )

    avg_support = sum(claim.support_score for claim in relevant_claims) / len(relevant_claims)

    unsupported = [
        claim.claim for claim in relevant_claims if claim.support_status == "unsupported"
    ]
    partial = [
        claim.claim for claim in relevant_claims if claim.support_status == "partial"
    ]

    problems: list[str] = []

    if unsupported:
        problems.append(f"Unsupported book-grounded claims: {unsupported}")

    if partial:
        problems.append(f"Partially supported book-grounded claims: {partial}")

    return metric_score(
        avg_support,
        f"Book-grounding average is {avg_support:.3f} across {len(relevant_claims)} relevant claims.",
        problems,
    )


def build_citation_alignment_metric(
    claim_scores: list[AnswerClaimScore],
) -> AnswerMetricScore:
    claims_requiring_sources = [
        claim
        for claim in claim_scores
        if claim.support_status in {"supported", "partial"}
        and claim.field_name
        in {
            "direct_answer",
            "book_summary",
            "key_points",
            "final_answer",
            "caveats",
        }
    ]

    if not claims_requiring_sources:
        return metric_score(
            0.0,
            "No supported or partially supported source-grounded claims were extracted.",
            ["No source-aligned claims could be evaluated."],
        )

    aligned_count = sum(
        1 for claim in claims_requiring_sources if claim.has_valid_source
    )

    score = aligned_count / len(claims_requiring_sources)

    weak_claims = [
        claim.claim
        for claim in claims_requiring_sources
        if not claim.has_valid_source
    ]

    problems: list[str] = []

    if weak_claims:
        problems.append(
            f"Supported/partial claims without valid retrieved chunk IDs: {weak_claims}"
        )

    return metric_score(
        score,
        f"{aligned_count}/{len(claims_requiring_sources)} source-grounded claims cited valid retrieved chunk IDs.",
        problems,
    )


def build_example_discipline_metric(
    claim_scores: list[AnswerClaimScore],
) -> AnswerMetricScore:
    problematic_claims = [
        claim
        for claim in claim_scores
        if claim.example_error_type != "none"
    ]

    if not claim_scores:
        return metric_score(
            0.0,
            "No claims were extracted, so example discipline cannot be checked.",
            ["No valid answer claims were extracted."],
        )

    if not problematic_claims:
        return metric_score(
            1.0,
            "No example-generalization errors were detected.",
            [],
        )

    score = max(0.0, 1.0 - (len(problematic_claims) / len(claim_scores)))

    problems = [
        f"{claim.example_error_type}: {claim.claim}"
        for claim in problematic_claims
    ]

    return metric_score(
        score,
        "Some claims turn a specific example, case study, or product result into a broader rule.",
        problems,
    )


def build_final_answer_consistency_metric(
    raw_score: object,
    reason: object,
    final_answer: str,
    key_points: list[str],
) -> AnswerMetricScore:
    score = clamp_score(raw_score)
    problems: list[str] = []

    # Simple deterministic guardrail: important key terms from key_points should
    # not disappear from final_answer.
    final_lower = final_answer.lower()
    lost_points: list[str] = []

    for point in key_points:
        point_lower = point.lower()
        # crude but useful: require at least one meaningful phrase/word from each point
        important_words = [
            word.strip(".,:;()[]{}").lower()
            for word in point_lower.split()
            if len(word.strip(".,:;()[]{}")) >= 7
        ]

        if important_words and not any(word in final_lower for word in important_words):
            lost_points.append(point)

    if lost_points:
        score = min(score, 0.75)
        problems.append(f"Final answer may have dropped key points: {lost_points}")

    return metric_score(score, str(reason), problems)


def build_completeness_guarded_score(
    raw_score: object,
    reason: object,
    missing_points: list[str],
) -> AnswerMetricScore:
    score = clamp_score(raw_score)

    if missing_points:
        score = min(score, 0.85)

    return metric_score(
        score,
        str(reason),
        [f"Missing points: {missing_points}"] if missing_points else [],
    )


def calculate_structured_answer_metrics(
    question: str,
    direct_answer: str,
    book_summary: str,
    key_points: list[str],
    practical_usage: str,
    design_takeaway: str,
    caveats: list[str],
    final_answer: str,
    chunks: list[dict[str, Any]],
    judge: dspy.Predict | None = None,
) -> StructuredAnswerMetricResult:
    if judge is None:
        judge = dspy.Predict(StructuredRAGAnswerJudge)

    retrieved_context = stringify_retrieved_context(chunks)
    retrieved_sources = stringify_retrieved_sources(chunks)
    retrieved_chunk_ids = get_retrieved_chunk_ids(chunks)

    prediction = judge(
        question=question,
        direct_answer=direct_answer,
        book_summary=book_summary,
        key_points=json.dumps(key_points, ensure_ascii=False),
        practical_usage=practical_usage,
        design_takeaway=design_takeaway,
        caveats=json.dumps(caveats, ensure_ascii=False),
        final_answer=final_answer,
        retrieved_context=retrieved_context,
        retrieved_sources=json.dumps(retrieved_sources, ensure_ascii=False),
    )

    claim_scores, claim_parse_problems = parse_claim_scores(
        raw_claims=getattr(prediction, "claim_scores_json", ""),
        retrieved_chunk_ids=retrieved_chunk_ids,
    )

    missing_points = normalize_list(getattr(prediction, "missing_points_json", "[]"))
    overall_problems = normalize_list(getattr(prediction, "overall_problems_json", "[]"))

    directness = metric_score(
        clamp_score(getattr(prediction, "directness_score", 0.0)),
        str(getattr(prediction, "directness_reason", "")),
    )

    book_grounding = build_book_grounding_metric(claim_scores)

    key_point_coverage = build_completeness_guarded_score(
        getattr(prediction, "key_point_coverage_score", 0.0),
        getattr(prediction, "key_point_coverage_reason", ""),
        missing_points,
    )

    practical_usefulness = metric_score(
        clamp_score(getattr(prediction, "practical_usefulness_score", 0.0)),
        str(getattr(prediction, "practical_usefulness_reason", "")),
    )

    design_takeaway_quality = metric_score(
        clamp_score(getattr(prediction, "design_takeaway_quality_score", 0.0)),
        str(getattr(prediction, "design_takeaway_quality_reason", "")),
    )

    caveat_quality = metric_score(
        clamp_score(getattr(prediction, "caveat_quality_score", 0.0)),
        str(getattr(prediction, "caveat_quality_reason", "")),
    )

    final_answer_consistency = build_final_answer_consistency_metric(
        getattr(prediction, "final_answer_consistency_score", 0.0),
        getattr(prediction, "final_answer_consistency_reason", ""),
        final_answer=final_answer,
        key_points=key_points,
    )

    citation_alignment = build_citation_alignment_metric(claim_scores)

    example_discipline = build_example_discipline_metric(claim_scores)

    clarity = metric_score(
        clamp_score(getattr(prediction, "clarity_score", 0.0)),
        str(getattr(prediction, "clarity_reason", "")),
    )

    final_score = (
        0.18 * book_grounding.score
        + 0.14 * key_point_coverage.score
        + 0.14 * practical_usefulness.score
        + 0.12 * design_takeaway_quality.score
        + 0.10 * final_answer_consistency.score
        + 0.10 * directness.score
        + 0.08 * citation_alignment.score
        + 0.06 * caveat_quality.score
        + 0.04 * example_discipline.score
        + 0.04 * clarity.score
    )

    claim_level_problems: list[str] = []

    for claim in claim_scores:
        if claim.problems:
            claim_level_problems.extend(
                [f"{claim.claim}: {problem}" for problem in claim.problems]
            )

    problems = (
        overall_problems
        + claim_parse_problems
        + book_grounding.problems
        + key_point_coverage.problems
        + practical_usefulness.problems
        + design_takeaway_quality.problems
        + final_answer_consistency.problems
        + citation_alignment.problems
        + example_discipline.problems
        + caveat_quality.problems
        + claim_level_problems
    )

    deduped_problems: list[str] = []
    seen: set[str] = set()

    for problem in problems:
        if problem not in seen:
            deduped_problems.append(problem)
            seen.add(problem)

    return StructuredAnswerMetricResult(
        question=question,
        direct_answer=direct_answer,
        book_summary=book_summary,
        key_points=key_points,
        practical_usage=practical_usage,
        design_takeaway=design_takeaway,
        caveats=caveats,
        final_answer=final_answer,
        retrieved_sources=retrieved_sources,
        directness=directness,
        book_grounding=book_grounding,
        key_point_coverage=key_point_coverage,
        practical_usefulness=practical_usefulness,
        design_takeaway_quality=design_takeaway_quality,
        caveat_quality=caveat_quality,
        final_answer_consistency=final_answer_consistency,
        citation_alignment=citation_alignment,
        example_discipline=example_discipline,
        clarity=clarity,
        claim_scores=claim_scores,
        missing_points=missing_points,
        final_score=round(final_score, 3),
        problems=deduped_problems,
    )