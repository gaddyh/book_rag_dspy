from pydantic import BaseModel, Field

from book_rag.core.models import BookChunk


class RetrieverMetricConfig(BaseModel):
    relevance_keywords: list[str] = Field(default_factory=list)
    bad_page_max: int = 30
    min_chunk_chars: int = 250


class RetrievedChunkScore(BaseModel):
    chunk_id: str
    page: int | None
    rank: int

    is_relevant: bool
    relevance_hits: list[str]

    is_bad_source: bool
    bad_source_reasons: list[str]

    text_preview: str


class RetrieverMetricResult(BaseModel):
    question: str
    k: int

    relevance_to_question: float
    source_quality: float

    relevant_chunk_count: int
    bad_chunk_count: int
    total_chunks: int

    chunk_scores: list[RetrievedChunkScore]

    final_score: float
    problems: list[str] = Field(default_factory=list)


def normalize(text: str) -> str:
    return text.lower()


def score_chunk_relevance(
    text: str,
    keywords: list[str],
) -> tuple[bool, list[str]]:
    normalized = normalize(text)

    hits = [
        keyword
        for keyword in keywords
        if keyword.lower() in normalized
    ]

    return len(hits) > 0, hits


def score_bad_source(
    chunk: BookChunk,
    bad_page_max: int,
    min_chunk_chars: int,
) -> tuple[bool, list[str]]:
    text = normalize(chunk.text)
    page = chunk.page_start

    reasons: list[str] = []

    if page is not None and page < bad_page_max:
        reasons.append(f"front_matter_page<{bad_page_max}")

    if "table of contents" in text:
        reasons.append("table_of_contents")

    if "copyright" in text:
        reasons.append("copyright")

    if "preface" in text and page is not None and page < bad_page_max:
        reasons.append("preface_front_matter")

    if len(text) < min_chunk_chars:
        reasons.append(f"too_short<{min_chunk_chars}")

    return len(reasons) > 0, reasons


def calculate_retriever_metrics(
    question: str,
    chunks: list[BookChunk],
    config: RetrieverMetricConfig,
) -> RetrieverMetricResult:
    chunk_scores: list[RetrievedChunkScore] = []

    for index, chunk in enumerate(chunks, start=1):
        text = chunk.text

        is_relevant, relevance_hits = score_chunk_relevance(
            text=text,
            keywords=config.relevance_keywords,
        )

        is_bad_source, bad_source_reasons = score_bad_source(
            chunk=chunk,
            bad_page_max=config.bad_page_max,
            min_chunk_chars=config.min_chunk_chars,
        )

        chunk_scores.append(
            RetrievedChunkScore(
                chunk_id=chunk.chunk_id,
                page=chunk.page_start,
                rank=index,
                is_relevant=is_relevant,
                relevance_hits=relevance_hits,
                is_bad_source=is_bad_source,
                bad_source_reasons=bad_source_reasons,
                text_preview=text[:250],
            )
        )

    total = len(chunk_scores)

    relevant_count = sum(1 for score in chunk_scores if score.is_relevant)
    bad_count = sum(1 for score in chunk_scores if score.is_bad_source)

    relevance_to_question = relevant_count / total if total else 0.0
    source_quality = 1.0 - (bad_count / total) if total else 0.0

    final_score = (
        0.65 * relevance_to_question
        + 0.35 * source_quality
    )

    problems: list[str] = []

    if relevant_count == 0:
        problems.append("No retrieved chunks matched relevance keywords.")

    if bad_count > 0:
        bad_sources = [
            score.chunk_id
            for score in chunk_scores
            if score.is_bad_source
        ]
        problems.append(f"Bad sources retrieved: {bad_sources}")

    weak_chunks = [
        score.chunk_id
        for score in chunk_scores
        if not score.is_relevant
    ]

    if weak_chunks:
        problems.append(f"Non-relevant chunks retrieved: {weak_chunks}")

    return RetrieverMetricResult(
        question=question,
        k=total,
        relevance_to_question=round(relevance_to_question, 3),
        source_quality=round(source_quality, 3),
        relevant_chunk_count=relevant_count,
        bad_chunk_count=bad_count,
        total_chunks=total,
        chunk_scores=chunk_scores,
        final_score=round(final_score, 3),
        problems=problems,
    )