# BookRAG DSPy — Evaluation-Driven RAG Engineering Lab

This project is a small but serious **RAG engineering lab** for book-based question answering.

The goal is not just to build a chatbot over a PDF.  
The goal is to make RAG behavior **measurable, inspectable, and improvable**.

It demonstrates an AI Engineering workflow:

```text
PDF → chunks → retrieval → context construction → DSPy answer generation → structured evaluation → experiment artifacts
```

The project currently focuses on answering questions over *30 Agents* and measuring answer quality across retrieval depths (k=5, k=10, k=20).

---

## Why this project exists

Most RAG demos stop at:

```text
retrieve chunks → ask LLM → print answer
```

That is not enough for production AI systems.

This repo explores the more important engineering question:

> **How do we know whether a RAG system is actually improving?**

To answer that, the system evaluates generated answers across several dimensions:

- directness
- book grounding
- key point coverage
- practical usefulness
- design takeaway quality
- caveat quality
- final answer consistency
- citation alignment
- example discipline
- clarity

It also performs claim-level grounding checks against retrieved sources.

---

## Current result snapshot

Latest structured answer evaluation run:

| Metric | Average |
|---|---|
| Final score | 0.946 |
| Directness | 0.944 |
| Book grounding | 0.989 |
| Key point coverage | 1.000 |
| Practical usefulness | 0.872 |
| Design takeaway quality | 0.967 |
| Caveat quality | 0.811 |
| Final answer consistency | 0.889 |
| Citation alignment | 1.000 |
| Example discipline | 1.000 |
| Clarity | 0.956 |

The run evaluated 9 configurations: 3 questions × 3 retrieval depths.

Best observed configurations:

| Question | Best k | Final score |
|---|---|---|
| support_001 | 10 | 0.974 |
| cognitive_loop_001 | 5 | 0.974 |
| coding_agent_001 | 10 | 0.950 |

**Important finding:**

More retrieved context is not automatically better.  
k=20 sometimes introduces extra unsupported or weakly cited claims, while k=5 or k=10 can produce cleaner answers.

---

## What this repo showcases

This repo is intended to showcase practical AI Engineering skills:

### 1. RAG system design

The system separates the core stages of a RAG pipeline:

- ingestion
- chunking
- retrieval
- context building
- answer generation
- evaluation
- artifact writing

This makes each layer easier to test, replace, and optimize.

### 2. Typed boundaries

The project is moving from loose dictionaries and script-style flow toward explicit domain objects.

Core concepts include:

- `BookChunk`
- `RetrievedContext`
- `StructuredEvalResult`

This makes the system easier to reason about and prepares it for controlled experiments.

### 3. Context construction as an experiment surface

The current context strategy is simple:

```text
ChunkOnlyContextBuilder
```

But the architecture is designed so future strategies can be compared:

- `ChunkOnlyContextBuilder`
- `PageWindowContextBuilder`
- `NeighborChunkContextBuilder`
- `ImageAwareContextBuilder`
- `CompressedContextBuilder`

The goal is to **measure** which context construction strategy produces the best grounded answer, not guess.

### 4. Structured answer evaluation

Generated answers are evaluated by an LLM judge using a structured metric schema.

Each answer receives scores for:

- `final_score`
- `directness`
- `book_grounding`
- `key_point_coverage`
- `practical_usefulness`
- `design_takeaway_quality`
- `caveat_quality`
- `final_answer_consistency`
- `citation_alignment`
- `example_discipline`
- `clarity`

The evaluator also stores explanations and problems for each score.

### 5. Claim-level grounding

The evaluator checks individual claims against retrieved chunks.

For each claim, it tracks:

- `claim`
- `field_name`
- `support_status`
- `support_score`
- `supporting_sources`
- `matched_source_ids`
- `has_valid_source`
- `source_alignment`
- `problems`

This allows the system to identify cases where an answer sounds good but contains claims that are not properly grounded.

### 6. Persistent experiment artifacts

The answer evaluation CLI writes durable artifacts for every run:

```text
reports/answer_eval/<run-name>/
  results.json
  summary.csv
  report.md
```

- `results.json` contains the full structured evaluation result.
- `summary.csv` contains a flat table for comparison.
- `report.md` contains a human-readable report with averages and per-run metrics.

This is the difference between a prompt demo and an evaluation-driven AI system.

---

## Project structure

```text
book_rag_dspy/
│
├── data/
│   └── processed/
│
├── evals/
│   └── answer evaluation datasets
│
├── reports/
│   └── answer_eval/
│       └── <run-name>/
│           ├── results.json
│           ├── summary.csv
│           └── report.md
│
├── src/
│   └── book_rag/
│       ├── chunker.py
│       ├── programs.py
│       ├── answer_metrics_llm.py
│       │
│       ├── answering/
│       │   └── rag_program.py
│       │
│       ├── context/
│       │   └── context_builder.py
│       │
│       ├── core/
│       │   └── models.py
│       │
│       ├── evaluation/
│       │   └── artifacts.py
│       │
│       ├── ingest/
│       │   └── ingest_pdf.py
│       │
│       └── retriever/
│           └── retriever.py
│
├── pyproject.toml
└── README.md
```

---

## Pipeline

### 1. Ingest the book

The ingestion step parses the source PDF and creates book chunks.

```text
PDF → BookChunk[]
```

Each chunk contains source metadata such as:

- `chunk_id`
- `text`
- `source`
- `page_start`
- `page_end`
- `has_images`
- `image_count`

After retrieval, chunks can also include:

- `rank`
- `score`
- `metadata`

### 2. Retrieve relevant chunks

The retriever searches for chunks relevant to a question.

```text
question → BookRetriever.search() → list[BookChunk]
```

The retriever returns typed `BookChunk` objects rather than raw dictionaries.

### 3. Build retrieved context

The context builder turns retrieved chunks into the text passed to the answer generator.

```text
list[BookChunk] → ChunkOnlyContextBuilder → RetrievedContext
```

`RetrievedContext` contains:

- `text`
- `chunks`
- `strategy`
- `metadata`

This boundary allows different context strategies to be tested later.

### 4. Generate an answer with DSPy

`BookRAG` orchestrates retrieval, context construction, and answer generation.

```text
question
  → retriever.search()
  → context_builder.build()
  → DSPy answer module
  → structured answer + citations
```

`BookRAG` is intentionally kept as orchestration logic, not a place for retrieval formatting or evaluation logic.

### 5. Evaluate answers

The answer evaluation runner tests multiple questions across multiple k values.

Example:

```text
support_001            k=5, k=10, k=20
cognitive_loop_001     k=5, k=10, k=20
coding_agent_001       k=5, k=10, k=20
```

The evaluator produces both metric-level and claim-level judgments.

### 6. Save artifacts

Every evaluation run writes:

- `results.json`
- `summary.csv`
- `report.md`

This creates a permanent record of system behavior.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

---

## Run answer evaluation

```bash
python -m book_rag.cli_answer_eval --run-name unstructured_fast_k_sweep
```

This creates:

```text
reports/answer_eval/unstructured_fast_k_sweep/
  results.json
  summary.csv
  report.md
```

---

## Example report output

```text
Runs evaluated: 9
Average final score: 0.946
Average book grounding: 0.989
Average citation alignment: 1.000
Average clarity: 0.956
```

Example comparison:

| Question ID | k | Final | Ground | Practical | Consistency | Citations |
|---|---|---|---|---|---|---|
| support_001 | 5 | 0.949 | 1.000 | 0.900 | 0.750 | 1.000 |
| support_001 | 10 | 0.974 | 1.000 | 0.900 | 1.000 | 1.000 |
| support_001 | 20 | 0.956 | 0.900 | 0.950 | 0.950 | 1.000 |
| cognitive_loop_001 | 5 | 0.974 | 1.000 | 0.900 | 1.000 | 1.000 |
| cognitive_loop_001 | 10 | 0.926 | 1.000 | 0.800 | 0.900 | 1.000 |
| cognitive_loop_001 | 20 | 0.941 | 1.000 | 0.850 | 0.750 | 1.000 |
| coding_agent_001 | 5 | 0.913 | 1.000 | 0.850 | 0.750 | 1.000 |
| coding_agent_001 | 10 | 0.950 | 1.000 | 0.900 | 0.900 | 1.000 |
| coding_agent_001 | 20 | 0.934 | 1.000 | 0.800 | 1.000 | 1.000 |

---

## Current engineering lesson

The first evaluation artifact shows that retrieval depth must be treated as a measurable parameter.

k=20 retrieves more context, but that does not guarantee better answers. Larger context can introduce extra claims, weaker focus, or citation problems.

This suggests that the next phase should compare context strategies, not simply increase k.

---

## Next experiments

Planned experiment directions:

### 1. Add best-config summary to reports

Generate a section showing the best k per question.

```text
question_id → best k → final score → grounding → consistency
```

### 2. Compare context builders

Add and evaluate:

- `ChunkOnlyContextBuilder`
- `PageWindowContextBuilder`
- `NeighborChunkContextBuilder`

**Goal:** Determine whether page-level or neighbor-expanded context improves grounding and practical usefulness without hurting consistency.

### 3. Add retrieval quality metrics

Measure retrieval independently from answer quality.

Possible retrieval metrics:

- source recall
- expected chunk hit rate
- page hit rate
- top-k source coverage

### 4. Improve practical usefulness

Current average practical usefulness is lower than grounding and citation alignment.

That suggests the system is well-grounded but could produce more actionable engineering guidance.

Possible improvements:

- better answer goal
- more explicit practical usage prompt
- context compression
- example-aware answer generation

### 5. Track experiment history

Add a lightweight experiment index:

```text
reports/answer_eval/index.md
```

This can summarize all runs over time.

---

## Why this matters

A production RAG system is not just a retriever plus an LLM.

It needs:

- clear data contracts
- controlled retrieval experiments
- context construction strategies
- answer quality metrics
- claim-level grounding
- persistent evaluation artifacts
- repeatable runs

This repo is a compact demonstration of that workflow.

---

## Status

Current status:

- ✅ PDF ingestion
- ✅ chunking
- ✅ retrieval
- ✅ DSPy answer generation
- ✅ structured answer evaluation
- ✅ claim-level source checks
- ✅ artifact writing
- ✅ k-sweep comparison
- 🚧 swappable context builder experiments
- 🚧 retrieval-specific metrics
- 🚧 report interpretation layer

---

## Project thesis

The central idea of this repo:

> Treat RAG as an engineering system that can be **measured and improved**, not as a prompt that either "feels good" or "feels bad."

The current implementation is intentionally small, but the workflow is the important part:

```text
build → evaluate → inspect → change one variable → evaluate again
```

That is the foundation of evaluation-driven AI engineering.
