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

## Standardization Strategy

One important research lesson from the benchmark work is that retrieval planning cannot be made robust by adding benchmark-specific heuristics alone.

The project therefore moved toward a more standardized design based on two principles:

1. dataset adapters may expose benchmark-specific structure
2. the planner and reranker should consume normalized evidence metadata

### Why Standardization Was Needed

T2-RAGBench contains richer source structure than UltraDomain.

For example, T2-RAGBench may provide:

- `table`
- `pre_text`
- `post_text`
- company name
- report year
- file name / page-level source information

UltraDomain does not expose the same fields in the same way.

If planning logic directly depends on benchmark-specific field names, then the system becomes difficult to generalize.

### Normalized Metadata Layer

To address this, a normalized evidence metadata layer was introduced so that different dataset adapters emit consistent planner-facing metadata.

Examples of normalized metadata fields now include:

- `source_title`
- `source_file`
- `source_domain`
- `source_subset`
- `source_split`
- `source_year`
- `source_entity`
- `source_page_number`
- `source_section`

This means the planner and reranker can reason over shared evidence features rather than raw dataset-specific keys.

### Why This Matters

This standardization allows the project to make a stronger research claim:

- the retrieval-planning framework is general
- dataset adapters expose whatever structure is available
- planning and reranking operate over normalized evidence capabilities

This is more defensible than building purely benchmark-specific retrieval rules.

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

### Experiment Group 3b: Structured T2-RAGBench Section Chunking

After further inspection of the T2-RAGBench source data, it became clear that the benchmark provides more structure than was being used correctly.

In particular, `FinQA` and `ConvFinQA` expose:

- `table`
- `pre_text`
- `post_text`

Originally, these were stored only as document-level metadata while chunking was still largely driven by flattened `context` text.

This caused an alignment problem:

- table chunks could inherit unrelated narrative fields
- narrative chunks could still carry the full table metadata
- field-aware retrieval and reranking were therefore noisier than intended

This was fixed by introducing section-aware T2 chunking.

The updated behavior now creates chunk structure more faithfully:

- `pre_text` -> narrative chunks
- `table` -> table chunks
- `post_text` -> narrative chunks

Chunk metadata is now aligned with actual section content instead of carrying all section fields indiscriminately.

This was a major structural improvement and was necessary before more meaningful table-aware retrieval experiments could be trusted.

### Experiment Group 4: Stronger Table-Aware Heuristic Reranking

The metadata-aware reranker was then strengthened to use chunk-level table signals directly.

Result:

- heuristic intent-aware planning started to improve slightly on T2-RAGBench
- gains were real but small and not always stable on larger samples

### Experiment Group 4b: Source-Aware Standardized Reranking

After the normalized metadata layer was introduced, reranking was extended to use standardized source-aware features such as:

- year cues
- entity/company cues
- title overlap
- file/source overlap
- page/source metadata

This was motivated by a frequent error pattern in T2-RAGBench:

- same company
- wrong year
- similar table/page structure
- high lexical overlap but incorrect source context

The standardized reranker now uses these normalized features across datasets rather than relying only on benchmark-specific logic.

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

## Compact Results Table

The table below summarizes the main experimental phases so far, what was changed, and what was learned.

| Phase | Benchmark / Setting | Main Change | Main Outcome | Interpretation |
|------|----------------------|-------------|--------------|----------------|
| 1 | UltraDomain baseline | Plain lexical retrieval with BM25 / keyword | BM25 established as a strong baseline | The retrieval baseline is meaningful and not artificially weak |
| 2 | UltraDomain early planning | Simple `document_aware` and `intent_document_aware` reranking | Almost no measurable gain | Coarse document-level heuristics alone were too weak |
| 3 | T2-RAGBench initial baseline | Plain BM25 over flattened mixed text-table context | Reasonable lexical retrieval, but strong table-heavy ambiguity | T2-RAGBench is a much harder and more relevant benchmark for mixed evidence retrieval |
| 4 | T2-RAGBench early planning | Initial rule-based planning over flattened chunks | Planning often hurt or had no effect | The dataset structure was not yet represented faithfully enough |
| 5 | T2-RAGBench diagnostics | Added chunk-level metadata and richer evaluation traces | Revealed that retrieval was already strongly table-oriented | The problem was not absence of tables, but weak use of evidence structure |
| 6 | T2-RAGBench table-aware chunking | Table-aware chunking and chunk-type metadata | Slight improvements began to appear | Chunk-level table/narrative structure matters for planning |
| 7 | T2-RAGBench stronger heuristic reranking | Table-aware chunk boosts and improved metadata-aware reranking | Small but real heuristic improvements on some samples | The core idea was not fundamentally wrong, but needed stronger signals |
| 8 | T2-RAGBench first small LLM planner | Unconstrained LLM retrieval planner | Much worse than heuristic and naive baselines | The LLM planner collapsed into overusing `statistical_preferred` |
| 9 | T2-RAGBench constrained LLM planner | Prompt constraints plus code-side guardrails | Much better than first LLM version, but still unstable overall | Small LLM planning is promising but not yet robust enough |
| 10 | Planner-controlled subset selection | Pre-retrieval chunk subset filtering before BM25 | Hard filtering often degraded retrieval | Candidate pruning was too aggressive and removed useful evidence |
| 11 | Oracle candidate recall analysis | Added `oracle_candidate_hit_rate` | Planning variants showed much higher candidate recall than naive BM25 | Candidate generation improved more than final ranking metrics suggested |
| 12 | Structured T2-RAGBench chunking | Separate chunking of `pre_text`, `table`, and `post_text` | More faithful benchmark representation | T2 structure needed to be aligned to chunks, not kept only at document metadata level |
| 13 | Normalized metadata standardization | Added shared fields like `source_year`, `source_entity`, `source_title` | Planning and reranking became less benchmark-specific | The framework became more general and better suited for future HoWie data |
| 14 | Source-aware heuristic reranking | Year-aware and entity-aware reranking using normalized metadata | Final top-k retrieval improved over earlier unstable versions | Same-company / wrong-year ambiguity is a major failure mode and can be addressed systematically |
| 15 | Improved heuristic planning (current strongest non-LLM stage) | Structured T2 chunking + normalized metadata + source-aware reranking | `document_aware` improved `Hit@1`, `Hit@5`, and `MRR@5`; `intent_document_aware` improved `Hit@5` and `MRR@5` over naive on the 250-sample | This is the strongest current evidence that the retrieval-planning idea works when benchmark structure is represented properly |

### Key Numbers From The Current Strongest Heuristic Setup

On the T2-RAGBench 250-question sample after structured T2 chunking, normalized metadata, and source-aware reranking:

| Variant | Hit@1 | Hit@5 | MRR@5 | Oracle Candidate Hit Rate | Interpretation |
|--------|--------|--------|--------|----------------------------|----------------|
| `naive` | 0.288 | 0.596 | 0.4061 | 0.596 | Strong lexical baseline |
| `document_aware` | 0.296 | 0.620 | 0.4235 | 0.796 | Best top-1 improvement among current heuristic variants |
| `intent_document_aware` | 0.288 | 0.632 | 0.4245 | 0.804 | Best final evidence-set quality and strongest candidate recall |

### Why These Results Matter

These results show an important pattern:

- naive BM25 remains a strong baseline
- planning substantially improves candidate recall
- after structural and source-aware fixes, planning also improves final retrieval quality
- `document_aware` and `intent_document_aware` are useful in different ways

This is a stronger and more convincing result than the earlier experiment phases where planning either had no effect or made retrieval worse.

### Evaluation Stability Requirement

An important implementation requirement for the project is that new retrieval ideas must be added without destroying older evaluation paths.

This is necessary because the final thesis comparison must still include:

- older baselines
- intermediate heuristic variants
- newer stronger planner-controlled retrieval variants

Therefore, new methods should be introduced as additional retriever or planning modes rather than rewriting the existing evaluation pipeline in place.

This ensures that the repo remains experimentally reproducible and that earlier results can still be rerun for thesis tables and ablations.

### Pre-Standardization T2-RAGBench Heuristic Results

Before introducing:

- structured T2 section-aware chunking
- normalized source metadata
- source-aware reranking

the strongest comparable heuristic planning run on the T2-RAGBench 250-question sample produced the following results:

| Variant | Hit@1 | Hit@5 | MRR@5 | Oracle Candidate Hit Rate | Interpretation |
|--------|--------|--------|--------|----------------------------|----------------|
| `naive` | 0.336 | 0.628 | 0.4447 | 0.628 | Strong lexical baseline over flattened mixed context |
| `document_aware` | 0.332 | 0.608 | 0.4363 | 0.796 | Candidate generation improved substantially, but final ranking quality dropped |
| `intent_document_aware` | 0.332 | 0.612 | 0.4395 | 0.800 | Candidate generation improved substantially, but final ranking still failed to exploit the stronger pool fully |

### Why The Pre-Standardization Results Matter

These earlier results were important because they exposed the structural weakness in the pipeline.

Interpretation:

- planning already increased oracle candidate recall strongly
- but final top-k quality did not improve enough
- this suggested the problem was not only the planner itself
- the benchmark structure and source alignment were not represented well enough yet

This was one of the main motivations for introducing the standardized metadata layer and the section-aware T2 chunking strategy.

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

### 3b. Structured benchmark fields must be aligned to chunk evidence

It is not enough to preserve fields such as `table`, `pre_text`, and `post_text` as document-level metadata.

Those fields need to be aligned to the actual chunks used for retrieval.

The T2-RAGBench experiments showed that a chunk can become misleading if it carries full-document table and narrative metadata simultaneously.

The section-aware T2 chunking fix was therefore not just an implementation refinement. It was a necessary correction in how benchmark structure was represented in the retrieval pipeline.

### 4. Heuristic intent-aware planning can help slightly

After chunk-level table signals were integrated into reranking, heuristic planning started to improve some results.

These gains were small, but they were the first evidence that the core idea was not fundamentally wrong.

### 4b. Candidate generation and final ranking must be evaluated separately

The addition of `oracle_candidate_hit_rate` produced one of the most important findings in the project.

On T2-RAGBench, planning variants often achieved much higher oracle candidate recall than naive BM25, even when final top-k metrics did not improve immediately.

This means:

- the planner was often finding the correct evidence somewhere in the candidate pool
- but the final reranking / selection step failed to convert that stronger candidate pool into better final retrieval

This is a critical methodological finding because it separates:

1. candidate generation quality
2. final ranking quality

Without oracle candidate recall, those failure modes would remain conflated.

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

### 7. Standardization improved the non-LLM heuristic path

After introducing:

- normalized evidence metadata
- structured T2 section-aware chunking
- source-aware reranking

the heuristic planning variants improved meaningfully compared with earlier unstable versions.

For example, on a T2-RAGBench 250-question sample, results reached:

- `naive`
  - `Hit@1 = 0.288`
  - `Hit@5 = 0.596`
  - `MRR@5 = 0.4061`
  - `Oracle candidate hit rate = 0.596`

- `document_aware`
  - `Hit@1 = 0.296`
  - `Hit@5 = 0.620`
  - `MRR@5 = 0.4235`
  - `Oracle candidate hit rate = 0.796`

- `intent_document_aware`
  - `Hit@1 = 0.288`
  - `Hit@5 = 0.632`
  - `MRR@5 = 0.4245`
  - `Oracle candidate hit rate = 0.804`

These results are important because they show:

- planning significantly improves candidate recall
- after structural and source-aware fixes, some of that gain is finally converted into final retrieval gains
- `document_aware` and `intent_document_aware` help in different ways

Interpretation:

- `document_aware` improved top-1 precision more clearly
- `intent_document_aware` improved final top-5 evidence coverage and ranking quality more clearly

This is the strongest non-LLM evidence so far that the retrieval-planning idea can work when the benchmark structure is represented properly and the reranker uses more than coarse document-level type information.

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
4. Compare the improved heuristic planner against semantic retrieval baselines once the current standardized lexical path stabilizes

## Recommended Experiment Families

To keep experiments reproducible and clearly grouped for thesis reporting, the retrieval-planning CLI now supports explicit variant families.

### Core Non-LLM Family

This is the historical and most stable comparison family.

- `naive`
- `document_aware`
- `intent_document_aware`

CLI group:

- `all`
- `all_core`

### Extended Non-LLM Family

This includes the stronger planner-controlled retrieval branch in addition to the core heuristic variants.

- `naive`
- `document_aware`
- `intent_document_aware`
- `intent_guided_retrieval`

CLI group:

- `all_non_llm`

### LLM Family

This is the comparison family for LLM-based planning variants.

- `llm_intent_document_aware`
- `llm_document_aware`
- `llm_guided_retrieval`

CLI group:

- `all_llm`

### Full Comparison Family

This runs all currently implemented retrieval-planning variants.

CLI group:

- `all_full`

## Recommended Evaluation Matrix

For T2-RAGBench, the preferred comparison structure is now:

### Representation Family A: Flat

- flat `naive`
- flat `document_aware`
- flat `intent_document_aware`
- optional flat LLM variants

### Representation Family B: Structured

- structured `naive`
- structured `document_aware`
- structured `intent_document_aware`
- optional structured LLM variants

This allows the thesis to separate:

1. the effect of representation (`flat` vs `structured`)
2. the effect of planning (`naive` vs heuristic vs LLM)

## Workflow Recommendation

For final thesis evaluation, experiments should be run as separate, clearly named families rather than one mixed command history.

Recommended sequence:

1. Prepare benchmark split explicitly, for example `test` only
2. Build flat chunk index
3. Build structured chunk index
4. Run `all_non_llm` on the flat index
5. Run `all_non_llm` on the structured index
6. Run `all_llm` on the flat index if needed
7. Run `all_llm` on the structured index if needed
8. Compare not only final retrieval metrics, but also oracle candidate recall

This organization makes it easier to write final thesis result tables and to reproduce all major comparisons later.

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
