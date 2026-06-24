# HoWieRAG Research Summary

## Purpose

This document is a clear research-oriented summary of what has been built, what has been tested, what was learned, why the current evaluation metrics are used, and what the next research steps should be.

It is intended to support:

- thesis writing
- experiment tracking
- design justification
- future implementation decisions

## Project Goal

HoWieRAG is a modular Retrieval-Augmented Generation (RAG) research system.

The current research focus is not just answer generation. The current main question is:

Can retrieval be improved by planning retrieval differently depending on the question type and the kind of source evidence available?

The guiding intuition is:

- some questions need precise factual lookup
- some need broader summary-style retrieval
- some need table or numeric evidence
- some need narrative explanation or contextual evidence

Therefore, a fixed retrieval strategy may not be optimal across all question types.

## Current System

The current pipeline is modular and consists of the following stages.

1. Query understanding
2. Document ingestion
3. Document classification
4. Chunking
5. Retrieval
6. Retrieval planning / reranking
7. Optional LLM-based answer generation

### Query Understanding

The repository includes a rule-based intent classifier with these labels:

- `FACT`
- `SUMMARY`
- `COMPARISON`
- `METHOD_CONTEXT`
- `LIMITATION`
- `TREND_PATTERN`
- `EXPLANATION`
- `SOURCE_SEEKING`
- `NAVIGATION`
- `INTERPRETATION`
- `DECISION_SUPPORT`
- `FOLLOWUP`
- `UNKNOWN`

This classifier is used both as a baseline query-understanding module and as part of retrieval planning.

### Document Ingestion

The system supports:

- standard file ingestion (`.txt`, `.md`, `.pdf`)
- dataset-style ingestion for benchmarks

Two benchmark adapters are now implemented:

- `UltraDomain`
- `T2-RAGBench`

### Document Classification

Documents are heuristically classified into:

- `narrative`
- `statistical`
- `mixed`

Additional metadata includes:

- `has_tables`
- `has_figures`

This classification is later used for retrieval planning.

### Chunking

The original chunking strategy was simple fixed-size character chunking.

It was later extended with table-aware behavior so that:

- table-like lines can be grouped together
- markdown-like tables are less likely to be split destructively
- chunk-level metadata is produced

Current chunk-level metadata includes:

- `chunk_type` (`narrative`, `table`, `mixed`)
- `has_table_like_content`
- `table_line_count`
- `table_line_ratio`

This was especially important for T2-RAGBench, where mixed text-table evidence is central.

### Retrieval

The current base retrievers are lexical:

- `keyword`
- `bm25`

No vector database is currently used.

The current system retrieves over chunk JSONL files stored on disk and loaded into memory. This is deliberate because it keeps the experimental setup simple and makes the effect of retrieval planning easier to isolate.

### Retrieval Planning

Three main retrieval-planning conditions have been tested:

1. `naive`
2. `document_aware`
3. `intent_document_aware`

Later, an additional condition was added:

4. `llm_document_aware`

The current planning logic works by reranking candidate chunks using:

- predicted question intent
- document metadata
- chunk-level table/narrative structure

## Benchmarks Used

### 1. UltraDomain

UltraDomain was used as the initial benchmark because it was already supported and provides a convenient context-to-question structure.

In the current setup:

- `context` is treated as the source document text
- `input` or `question` is treated as the query
- `answers` are the benchmark answers

UltraDomain was useful for building the baseline pipeline and early retrieval-planning experiments.

### 2. T2-RAGBench

T2-RAGBench is highly relevant to the thesis because it is explicitly designed for RAG evaluation over mixed text-and-table financial documents.

It includes subsets such as:

- `FinQA`
- `ConvFinQA`
- `TAT-DQA`

This benchmark is important because it is closer to the intended research question than ordinary QA-with-given-context datasets.

Why it matters:

- it evaluates retrieval, not just answer generation
- it contains mixed narrative and table evidence
- many questions are numeric, comparative, or table-oriented
- it makes table-aware retrieval planning more meaningful

## Why These Metrics Are Used

The primary evaluation metrics currently used are:

- `Hit@1`
- `Hit@5`
- `MRR@5`
- `Precision@1`

These are retrieval metrics, not answer-generation metrics.

This is intentional because the research question is currently about retrieval quality and retrieval planning.

### Hit@1

Definition:

- whether the top-ranked retrieved chunk is correct

Why it matters:

- this is the cleanest measure of ranking quality at the top position
- if retrieval planning is useful, it should ideally improve the chance that the very first chunk is the right one
- this matters especially for RAG because the top-ranked chunk strongly influences downstream answer generation

Interpretation:

- higher `Hit@1` means better first-result precision
- useful for testing whether planning improves the ranking order, not just recall

### Hit@5

Definition:

- whether a correct chunk appears anywhere in the top 5 retrieved chunks

Why it matters:

- many RAG systems do not rely on a single chunk
- if the correct evidence is somewhere in the retrieved set, an answer model may still succeed
- this metric is more forgiving than `Hit@1`

Interpretation:

- higher `Hit@5` means retrieval is finding the right evidence somewhere in the retrieval set
- if `Hit@5` is high but `Hit@1` is lower, reranking may be a promising direction

### MRR@5

Definition:

- Mean Reciprocal Rank over the top 5
- if the first correct chunk appears at rank 1, score is `1.0`
- if at rank 2, score is `0.5`
- if at rank 3, score is `0.333...`
- and so on

Why it matters:

- it captures ranking quality more smoothly than Hit@1 or Hit@5
- it rewards correct evidence appearing earlier
- it is useful when planning moves the right chunk from lower rank to higher rank, even if `Hit@5` stays unchanged

Interpretation:

- a higher `MRR@5` means the correct evidence is appearing earlier on average

### Precision@1

Definition:

- top-result correctness in the current setup

Why it matters:

- it is effectively another top-1 quality measure in this benchmark design
- useful for reporting top-position precision alongside Hit@1

## Why Retrieval Metrics Are Important Before Answer Metrics

The current evaluation emphasizes retrieval first because:

- if retrieval is weak, answer generation will be weak regardless of the LLM
- retrieval planning should be tested independently from answer generation
- otherwise it becomes difficult to tell whether gains came from better retrieval or from the answer model compensating for poor retrieval

This is methodologically cleaner for the thesis.

## Main Research Question So Far

The current core question has been:

Can retrieval be improved by adapting retrieval behavior based on:

- query intent
- source/document characteristics
- chunk-level table vs narrative structure

## Main Experiments Conducted So Far

### Experiment Group 1: Baseline Retrieval

The first experiments established a lexical retrieval baseline.

The strongest baseline so far is:

- `bm25 + naive`

This is the main baseline against which planning methods are compared.

### Experiment Group 2: Rule-Based Retrieval Planning

Tested conditions:

- `naive`
- `document_aware`
- `intent_document_aware`

Early results showed:

- on UltraDomain, simple heuristic planning had almost no measurable effect
- on early T2-RAGBench experiments, heuristic planning sometimes hurt performance

This suggested the original planning logic was too weak or too coarse.

### Experiment Group 3: Table-Aware Chunking and Chunk Diagnostics

After diagnosing failures, the system was improved by:

- making chunking more table-aware
- adding chunk-level structure metadata
- enriching evaluation outputs with debug fields such as:
  - `chunk_type`
  - `has_table_like_content`
  - `table_line_ratio`
  - `document_type`
  - `source_file`
  - `is_correct`

This made it possible to inspect whether the system was actually retrieving table-heavy chunks.

### Experiment Group 4: Stronger Table-Aware Heuristic Reranking

The metadata-aware reranker was then strengthened to use chunk-level table signals directly.

Result:

- heuristic intent-aware planning started to improve slightly on T2-RAGBench
- gains were real but small and not always stable on larger samples

### Experiment Group 5: Small LLM Planner

A small LLM-based retrieval planner was introduced.

The planner:

- predicts a retrieval intent
- chooses a retrieval mode
- chooses preferred document and chunk types
- returns a structured `RetrievalPlan`

The first unconstrained version performed badly because it collapsed almost all questions into table-heavy `statistical_preferred` routing.

This led to the addition of:

- prompt constraints
- validation
- code-side guardrails based on query cues

The constrained LLM planner improved substantially over the first LLM version.

## Findings So Far

### 1. BM25 is a strong baseline

Across the experiments, BM25 has been a strong lexical retriever.

This is important because it means the thesis is not built on an artificially weak baseline.

### 2. Simple heuristic planning is not enough by itself

Early rule-based planning often:

- had no effect
- or slightly hurt retrieval

This showed that coarse document-level heuristics alone were not enough.

### 3. Table-aware chunking and chunk-level structure matter

The diagnostics showed that T2-RAGBench retrieval is heavily table-driven.

Therefore:

- chunk-level table structure matters
- document-level type labels alone are too coarse

### 4. Heuristic intent-aware planning can help slightly

After chunk-level table signals were integrated into reranking, heuristic planning started to improve some results.

These gains were small, but they were the first evidence that the core idea was not fundamentally wrong.

### 5. The LLM planner can help specific intent groups

The constrained LLM planner showed strong gains for some intent categories, especially on smaller-sample analysis.

Promising categories included:

- `COMPARISON`
- `DECISION_SUPPORT`
- `EXPLANATION`
- some `FACT` questions

This suggests the LLM planner may be especially useful for semantically harder routing decisions.

### 6. The LLM planner is not yet the best overall method

Although the constrained LLM planner improved greatly compared with the first version, it has not yet consistently beaten the best heuristic planner overall.

This means:

- the small LLM is promising
- but not yet fully calibrated
- and likely needs either more constraints or hybrid use

## Why the Current Results Are Still Valuable

The current findings are useful even when the gains are small.

The experiments already support several meaningful conclusions:

1. retrieval planning is a real and testable problem
2. benchmark type matters
3. table-heavy finance benchmarks expose weaknesses that general heuristics miss
4. chunk-level evidence structure is important
5. unconstrained LLM planning can hurt strong lexical retrieval
6. constrained LLM planning can recover and improve some intent groups

This is already a valid and defensible research contribution trajectory.

## Current Thesis Direction

The strongest thesis direction currently appears to be:

Query-aware and evidence-aware retrieval planning for RAG, especially in mixed text-and-table settings.

A strong thesis framing would be:

- compare fixed lexical retrieval against heuristic retrieval planning and constrained LLM retrieval planning
- test across multiple benchmarks
- analyze where planning helps and where it hurts
- study the role of table-aware chunking and evidence-type routing

## Recommended Future Work

### Short-Term Next Steps

1. Continue diagnostic analysis by intent group and benchmark
2. Improve planner calibration rather than only adding more boosts
3. Add a hybrid planner variant

### Recommended Hybrid Planner Idea

A promising next design is:

- use heuristic planning for stable/common cases
- use LLM planning for harder semantic cases such as:
  - `COMPARISON`
  - `DECISION_SUPPORT`
  - `EXPLANATION`

This is motivated by the current findings that the LLM planner seems strongest on some semantic categories but not yet best overall.

### Medium-Term Next Steps

1. Evaluate the planner on larger samples and full benchmark runs
2. Add benchmark-specific routing analysis
3. Add answer-generation evaluation after retrieval is stable

### Longer-Term Next Steps

1. More explicit table-aware chunking refinements
2. Better table vs narrative cue detection
3. Answer evaluation against numeric gold answers on T2-RAGBench
4. Hybrid or selective LLM planning

## Practical Current Recommendation

At the current stage of the project:

- keep `bm25 + naive` as the main baseline
- keep the heuristic planner as the main non-LLM planning baseline
- keep the constrained small-LLM planner as an experimental condition
- prioritize hybrid planning and deeper planner calibration next

## Summary

The project has moved from a minimal RAG prototype to a real retrieval-planning research framework.

The most important lessons so far are:

- retrieval quality must be measured independently from answer generation
- BM25 is a strong baseline and should be treated seriously
- simple planning rules are often too weak when evidence is mixed and table-heavy
- chunk-level table structure matters
- constrained LLM planning is far more promising than unconstrained LLM planning
- the most promising next step is likely a hybrid retrieval planner rather than purely heuristic or purely LLM-based routing

This means the project is already producing meaningful research findings, even before final answer-generation experiments are fully completed.
