# File Structure Detail

## Overview

This project is an early research prototype for a modular RAG chatbot. Right now, the implemented stage is intent classification.

## Project Goal

The goal of `howie-rag` is to become a modular retrieval-augmented generation system for exploring higher-education and science-research study results.

The project is being built step by step rather than as one large final system. The idea is to keep each module isolated enough that it can be:

- implemented cleanly
- tested independently
- compared experimentally against alternative approaches
- improved later without rewriting the whole stack

This repository is therefore both:

- a software prototype
- a research experiment framework

## Current Stage

The current stage is query understanding through intent classification.

Right now, the system does not yet retrieve evidence or generate grounded answers. Instead, it tries to understand what kind of question the user is asking before later RAG steps are added.

This matters because different question types will likely need different downstream handling.

Examples:

- `FACT` may need precise retrieval of a specific result or statistic.
- `SUMMARY` may need broad retrieval across multiple passages.
- `SOURCE_SEEKING` may need provenance-heavy output with citations.
- `DECISION_SUPPORT` may need evidence synthesis plus caution.
- `FOLLOWUP` may require previous conversational context.

## Planned Larger System

The longer-term system will likely grow into several modules.

Expected future stages include:

1. Query understanding
   - intent classification
   - follow-up detection
   - possibly query rewriting or reformulation
2. Retrieval planning
   - decide what retrieval strategy fits the intent
3. Document retrieval
   - lexical, semantic, or hybrid retrieval
4. Evidence selection
   - choose the most relevant passages or chunks
5. Answer generation
   - generate a grounded answer from retrieved evidence
6. Citation and provenance handling
   - show where the answer came from
7. Evaluation
   - evaluate intent quality, retrieval quality, and answer quality

Not all of these parts exist yet. The repository is intentionally focused on building the pipeline one minimal stage at a time.

The next implemented stages after intent classification are minimal document loading, minimal chunking, minimal retrieval, a minimal end-to-end chatbot flow, and a first LLM-backed RAG answer stage.

The current system classifies a user question into one of these labels:

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

The project does not yet include vector databases, API layers, or UI layers.

## How The Current Intent Classifier Works

The rule-based classifier works in a simple, deterministic way:

1. Take the input question as text.
2. Convert it to lowercase.
3. Check it against keyword groups assigned to each intent.
4. Use a fixed priority order so overlaps resolve consistently.
5. Return an `IntentResult` with:
   - `intent`
   - `confidence`
   - `reasoning`
6. If nothing matches, return `UNKNOWN` with low confidence.

Example:

```bash
python scripts/classify_intent.py "What are the key findings?"
```

Expected output:

```text
Intent: SUMMARY
Confidence: 0.9
Reasoning: Matched keyword: 'key findings'
```

## Current Intent Definitions

### `FACT`

Specific factual detail.

Example: `What is the sample size?`

### `SUMMARY`

Overview or key findings.

Example: `Summarize the main findings`

### `COMPARISON`

Compare studies, methods, results, or groups.

Example: `How does study A differ from study B?`

### `METHOD_CONTEXT`

Methodology, participants, design, or dataset context.

Example: `How was the data collected?`

### `LIMITATION`

Caveats, bias, uncertainty, or weaknesses.

Example: `What are the limitations of this study?`

### `TREND_PATTERN`

Questions about trends, trajectories, changes, or patterns.

Example: `What trend do we see in enrollment over time?`

### `EXPLANATION`

Questions asking why or how something happened, or what something means.

Example: `Why did mobility rates decrease?`

### `SOURCE_SEEKING`

Questions asking where the answer came from, including citation or study source.

Example: `Which study reported this?`

### `NAVIGATION`

Questions asking where in the corpus or report to look.

Example: `Where can I find the section on student mobility?`

### `INTERPRETATION`

Questions asking for implications, analytical meaning, or reasoned reading.

Example: `What do these findings imply for policy?`

### `DECISION_SUPPORT`

Questions asking what action to take or what is preferable.

Example: `Which intervention seems most effective to adopt?`

### `FOLLOWUP`

Questions that depend on prior conversational context.

Example: `What about in rural schools?`

### `UNKNOWN`

Fallback label when no intent matches clearly.

## Priority Order In The Current Rule-Based Classifier

The classifier currently checks intents in this order:

1. `FOLLOWUP`
2. `COMPARISON`
3. `DECISION_SUPPORT`
4. `SOURCE_SEEKING`
5. `NAVIGATION`
6. `LIMITATION`
7. `INTERPRETATION`
8. `METHOD_CONTEXT`
9. `EXPLANATION`
10. `TREND_PATTERN`
11. `SUMMARY`
12. `FACT`
13. `UNKNOWN`

This order matters because one question can match multiple keyword groups.

Example:

- `What is the sample size?`
- `sample` looks like `METHOD_CONTEXT`
- `sample size` should be `FACT`

To avoid the broader method rule swallowing the fact query, the classifier includes a small special-case check for `sample size`.

## File By File

### `src/howie_rag/core/schemas.py`

Defines the main Pydantic data models.

Current models:

1. `IntentResult`
   - classifier output structure
   - fields:
     - `intent: str`
     - `confidence: float`
     - `reasoning: Optional[str]`
2. `Document`
   - placeholder for future document-level data
3. `Chunk`
   - placeholder for future chunk-level data

### `src/howie_rag/core/utils.py`

Contains `stable_id(text: str) -> str`.

It hashes text with SHA-256 and returns the first 12 characters. This is useful for deterministic short IDs.

### `src/howie_rag/intent/intent_labels.py`

Defines the allowed intent labels as the `IntentLabel` enum.

This centralizes the taxonomy so tests and classifiers use the same label set.

### `src/howie_rag/intent/base.py`

Defines the abstract classifier interface:

```python
classify(question: str) -> IntentResult
```

This is important because it lets the project compare multiple classifier implementations later, such as rule-based, ML-based, or LLM-based classifiers.

### `src/howie_rag/intent/rule_based.py`

Implements `RuleBasedIntentClassifier`.

This is the current working classifier. It uses:

- case-insensitive matching
- keyword groups by intent
- fixed priority resolution
- deterministic outputs
- simple reasoning strings showing the matched keyword

Direct matches return confidence `0.9`. Unmatched inputs return `UNKNOWN` with confidence `0.2`.

### `src/howie_rag/intent/__init__.py`

Exports the classifier classes for cleaner imports.

### `src/howie_rag/ingestion/__init__.py`

Exports the minimal document-loading entrypoint.

### `src/howie_rag/ingestion/text_loader.py`

Implements the first ingestion step.

This file currently loads `.txt`, `.md`, and `.pdf` files from a directory and converts them into `Document` objects.

For each supported file it creates:

- `doc_id`
- `title`
- `text`
- `metadata`

For PDF files, the current implementation performs minimal text extraction and inserts simple page markers such as `[Page 1]` when text is available.

The metadata currently stores:

- `source_path`
- `file_type`

It now also stores a first heuristic document classification, including:

- `document_type`
- `has_tables`
- `has_figures`
- `classification_stats`

This is intentionally small. It gives the project a clean, testable way to move from raw files on disk into structured documents before later chunking and retrieval are added.

### `src/howie_rag/datasets/__init__.py`

Exports dataset-style adapters and dataset record schemas.

### `src/howie_rag/datasets/schemas.py`

Defines two separate dataset concepts:

1. `SourceDocumentRecord`
   - used for source corpus ingestion and retrieval
2. `BenchmarkQARecord`
   - used for evaluation question/answer benchmark entries

This separation is important for datasets like UltraDomain where the same JSONL row contains both source text and QA information.

### `src/howie_rag/datasets/ultradomain.py`

Implements the UltraDomain dataset adapter.

It supports reading one `.jsonl` file or a folder of `.jsonl` files and can produce:

- source records from the `context` field
- benchmark QA records from `input` or `question` plus `answers`

The loader is defensive and currently:

- falls back from `input` to `question`
- falls back from `context_id` to a stable ID generated from the context text
- normalizes `answers` into a list
- preserves `meta` fields such as title and authors when available
- skips benchmark QA rows that do not contain valid answers

It also includes an adapter that converts UltraDomain source records into normal `Document` objects so they can flow through the existing classification, chunking, and retrieval pipeline without changing the normal TXT/MD/PDF/GovData ingestion path.

### `src/howie_rag/evaluation/ultradomain_retrieval.py`

Implements retrieval-focused benchmark evaluation for UltraDomain.

For each benchmark question it currently:

- runs the existing intent classifier
- runs the selected retriever over UltraDomain chunks
- checks whether any retrieved chunk matches the gold `doc_id` or `context_id`
- computes retrieval metrics including:
  - `Hit@1`
  - `Hit@5`
  - `MRR@5`
  - `Precision@1`

This keeps the first UltraDomain benchmark stage focused on retrieval correctness before answer-quality evaluation is added.

It now also supports retrieval-planning experiment variants:

- `naive`
- `document_aware`
- `intent_document_aware`

and reports:

- `Hit@1`
- `Hit@5`
- `MRR@5`
- `Precision@1`
- average retrieved chunks per question

### `src/howie_rag/retrieval_planning/__init__.py`

Exports the retrieval-planning experiment entrypoints.

### `src/howie_rag/retrieval_planning/retrieval_plan_schema.py`

Defines `RetrievalPlan`, which captures:

- the original query
- detected intent
- retrieval mode
- preferred document types
- metadata preferences
- top-k settings
- candidate pool size
- explanation of the current retrieval plan

### `src/howie_rag/retrieval_planning/rule_based_planner.py`

Implements a rule-based retrieval planner using the existing intent labels.

This planner maps question intent to retrieval preferences such as:

- narrative-preferred retrieval
- statistical-preferred retrieval
- source-metadata-preferred retrieval
- broad hybrid retrieval

### `src/howie_rag/retrieval_planning/metadata_aware_retriever.py`

Implements metadata-aware reranking.

The current version does not hard-filter chunks. Instead, it:

- retrieves a larger candidate pool
- boosts or reranks chunks based on document metadata
- preserves both original score and adjusted score for debugging

### `src/howie_rag/retrieval_planning/experiment_variants.py`

Implements the three retrieval experiment variants:

1. `naive`
2. `document_aware`
3. `intent_document_aware`

This is the core thesis experiment layer for comparing whether document type and question intent improve retrieval quality.

### `src/howie_rag/document_classification/__init__.py`

Exports the heuristic document classification entrypoint.

### `src/howie_rag/document_classification/heuristic_classifier.py`

Implements the first document-level classifier.

It currently classifies documents into:

- `narrative`
- `statistical`
- `mixed`

and also sets flags such as:

- `has_tables`
- `has_figures`

The current classifier is heuristic-based and uses simple structural signals such as:

- digit density
- table-like line ratio
- heading-like line ratio
- paragraph-like line ratio
- keyword hits for narrative, figure, and table patterns

This module is the first step toward structure-aware chunking and source-aware retrieval.

### `src/howie_rag/chunking/__init__.py`

Exports the minimal chunking entrypoints.

### `src/howie_rag/chunking/text_chunker.py`

Implements the first chunking step.

This file currently converts `Document` objects into smaller `Chunk` objects using a simple fixed-size character window with configurable overlap.

The current chunker:

- takes a `Document`
- splits its text into chunks of a chosen size
- optionally overlaps neighboring chunks
- preserves `doc_id`
- adds chunk metadata such as:
  - `title`
  - `chunk_index`
  - `start_index`
  - `end_index`

This is intentionally basic. It is meant to provide a working bridge between ingestion and retrieval before more advanced chunking strategies are explored.

### `src/howie_rag/retrieval/__init__.py`

Exports the minimal retrieval entrypoints.

### `src/howie_rag/retrieval/keyword_retriever.py`

Implements the first retrieval step.

This file currently retrieves `Chunk` objects using a simple lexical overlap score between the user query and chunk text.

The current retriever:

- tokenizes the query and chunk text case-insensitively
- scores each chunk by token overlap
- drops zero-score chunks
- sorts matches by score
- returns the top `k` matches

It returns `RetrievalMatch` objects containing:

- the matched `Chunk`
- the integer overlap `score`

This is intentionally simple and easy to debug. It gives the project a working retrieval layer before adding TF-IDF, embeddings, reranking, or vector search.

### `src/howie_rag/chatbot/__init__.py`

Exports the minimal chatbot pipeline entrypoint.

### `src/howie_rag/chatbot/pipeline.py`

Implements the first end-to-end chatbot flow.

This file currently wires together:

- intent classification
- document loading
- chunking
- retrieval

The main function is `run_simple_chatbot(...)`.

It returns a `ChatbotResponse` containing:

- the original question
- predicted intent
- confidence
- reasoning
- retrieved chunk matches

This is not yet a full answer-generation system. It is a working pipeline that shows how a user question moves through the first modular stages of the chatbot.

### `src/howie_rag/llm/__init__.py`

Exports the LLM client interfaces.

### `src/howie_rag/llm/base.py`

Defines the abstract LLM interface used by the app.

This is what keeps the project modular. The rest of the pipeline can call a generic LLM client without depending directly on one specific backend.

### `src/howie_rag/llm/vllm_client.py`

Implements the first concrete LLM backend.

This client talks to a local vLLM server through its OpenAI-compatible HTTP API.

It currently:

- sends a system prompt and user prompt
- sends the configured model name
- passes generation settings such as temperature and max tokens
- returns the generated assistant text

### `src/howie_rag/answering/__init__.py`

Exports the RAG answer-generation helpers.

### `src/howie_rag/answering/rag_answerer.py`

Implements the first grounded answer-generation layer.

This file currently:

- builds a prompt from:
  - the question
  - predicted intent
  - retrieved chunks
- adds a grounded system prompt
- sends that prompt to the configured LLM client
- returns the generated answer text

This is the first module that turns retrieval output into a real answer instead of only returning matching chunks.

### `src/howie_rag/intent/evaluation.py`

Provides the minimal evaluation pipeline for intent classification experiments.

This file currently handles:

- loading the CSV dataset into structured examples
- filtering by split (`train`, `dev`, `test`, or all)
- running a classifier over all examples
- computing accuracy, macro precision, macro recall, macro F1
- computing per-label precision, recall, F1, and support
- building and formatting a confusion matrix

This module is intentionally generic so that later classifiers can be evaluated through the same code.

### `scripts/classify_intent.py`

Small CLI script for manual testing.

Usage:

```bash
python scripts/classify_intent.py "What do these findings imply for policy?"
```

It prints:

- `Intent:`
- `Confidence:`
- `Reasoning:`

### `scripts/evaluate_rule_based_intent.py`

Small CLI script for benchmarking the current rule-based classifier against the dataset.

Default behavior:

```bash
python scripts/evaluate_rule_based_intent.py
```

This runs the `RuleBasedIntentClassifier` on the default `v3` dataset test split and prints:

- total examples
- accuracy
- macro precision / recall / F1
- per-label metrics
- confusion matrix

It can also be run with an explicit dataset path and split.

### `scripts/run_simple_chatbot.py`

Small CLI script for running the current end-to-end pipeline.

Usage:

```bash
python scripts/run_simple_chatbot.py "What trend do we see in student mobility?" path/to/documents
```

It currently prints:

- the input question
- predicted intent
- confidence
- reasoning
- retrieved chunk matches

### `scripts/run_rag_chatbot.py`

Small CLI script for running the current LLM-backed RAG flow.

Usage:

```bash
python scripts/run_rag_chatbot.py "What trend do we see in student mobility?" path/to/documents
```

It currently:

- runs the existing pipeline
- sends retrieved context to the local vLLM server
- prints the final generated answer

Configuration is environment-based:

- `HOWIE_LLM_BASE_URL`
- `HOWIE_LLM_MODEL`

### `scripts/prepare_ultradomain.py`

Reads one UltraDomain `.jsonl` file or a folder of domain files and writes:

- `source_documents.jsonl`
- `benchmark_questions.jsonl`

It also deduplicates repeated contexts.

### `scripts/build_ultradomain_index.py`

Loads the prepared UltraDomain source documents, converts them into normal `Document` objects, classifies them, chunks them, and writes:

- `documents_classified.jsonl`
- `chunks.jsonl`

### `scripts/evaluate_ultradomain_retrieval.py`

Loads prepared benchmark questions and built chunks, runs retrieval with a selected retriever (`keyword` or `bm25`), computes retrieval metrics, and saves results to:

- JSON
- CSV

### `scripts/evaluate_ultradomain_retrieval_planning.py`

Runs the retrieval-planning experiments over UltraDomain.

It supports:

- `--retriever keyword|bm25`
- `--variant naive|document_aware|intent_document_aware|all`
- `--top-k`
- `--candidate-pool-size`

It writes:

- one JSON result file per variant
- one per-question debug JSONL per variant
- one comparison CSV across variants

### `tests/test_project_setup.py`

Basic project smoke tests.

It currently checks:

1. `stable_id()` is deterministic
2. `IntentResult` can be created

### `tests/test_intent_rule_based.py`

Tests the current rule-based intent classifier.

It now covers examples for:

- `SUMMARY`
- `COMPARISON`
- `NAVIGATION`
- `FACT`
- `LIMITATION`
- `METHOD_CONTEXT`
- `TREND_PATTERN`
- `EXPLANATION`
- `SOURCE_SEEKING`
- `INTERPRETATION`
- `DECISION_SUPPORT`
- `FOLLOWUP`
- `UNKNOWN`

These tests verify that the intended labels work and that the rule ordering behaves as expected.

### `tests/test_intent_evaluation.py`

Tests the evaluation utilities.

It currently checks:

1. the dataset loader correctly filters the `v3` CSV by split
2. the metric calculations behave as expected on a small stub classifier example

### `tests/test_text_loader.py`

Tests the new minimal ingestion module.

It currently checks:

1. only supported `.txt` and `.md` files are loaded
2. unsupported files are ignored
3. document titles and metadata are populated correctly
4. document IDs remain stable across repeated loads
5. PDF files are accepted and their extracted text is loaded into `Document` objects

### `tests/test_document_classifier.py`

Tests the new heuristic document classifier.

It currently checks:

1. narrative-like text is classified as `narrative`
2. table-heavy text is classified as `statistical`
3. mixed report-and-table content is classified as `mixed`
4. figure-related text can trigger `has_figures`

### `tests/test_ultradomain_loader.py`

Tests the UltraDomain adapter.

It currently checks:

1. source records can be loaded from a single JSONL file
2. repeated contexts collapse into one source document record
3. QA records normalize string answers into lists
4. rows without answers are skipped for benchmark loading
5. when label is missing, the filename is used as the domain

### `tests/test_ultradomain_evaluation.py`

Tests the UltraDomain document adapter and retrieval evaluation path.

It currently checks:

1. UltraDomain source records can be converted to normal `Document` objects with preserved metadata
2. chunking preserves UltraDomain metadata required for evaluation
3. retrieval evaluation computes the expected `Hit@1`, `Hit@5`, `MRR@5`, and `Precision@1`

### `tests/test_retrieval_planning.py`

Tests the retrieval-planning layer.

It currently checks:

1. the rule-based planner maps intents to the expected retrieval modes
2. metadata-aware reranking boosts preferred document types
3. missing metadata does not break reranking
4. `naive`, `document_aware`, and `intent_document_aware` behave differently as intended

### `tests/test_text_chunker.py`

Tests the new minimal chunking module.

It currently checks:

1. text is split into multiple chunks with the expected overlap
2. empty input produces no chunks
3. multiple documents can be chunked together
4. invalid chunking parameters raise errors

### `tests/test_keyword_retriever.py`

Tests the new minimal retrieval module.

It currently checks:

1. the best matching chunks are returned first
2. matching is case-insensitive
3. zero-score chunks are excluded
4. invalid `top_k` values raise errors

### `tests/test_chatbot_pipeline.py`

Tests the new end-to-end chatbot assembly logic.

It currently checks:

1. the pipeline returns an intent plus retrieval matches for a relevant query
2. the pipeline handles queries with no retrieval matches cleanly

### `tests/test_rag_answerer.py`

Tests the first LLM-backed answer-generation layer.

It currently checks:

1. the RAG prompt contains the question, intent, and source chunks
2. the answerer calls the LLM client with the expected parameters

## Why This Structure Is Useful

This setup already supports an experiment comparing different classifiers because the interface is modular.

Later, the repository can add:

- `RuleBasedIntentClassifier`
- `MLIntentClassifier`
- `LLMIntentClassifier`

and evaluate all of them through the same `BaseIntentClassifier` interface.

This means intent classification is now set up as an actual experiment rather than just a one-off script.

The intended comparison is:

- `RuleBasedIntentClassifier`
- `MLIntentClassifier`
- `LLMIntentClassifier`

All three should be evaluated on the same dataset and splits.

## Current Limitations

The current classifier is intentionally simple.

It is useful for:

- clear phrasing
- easy debugging
- reproducible baselines
- fast prototyping

It is not foolproof because it still struggles with:

- paraphrases not covered by keywords
- spelling variation
- ambiguous wording
- overlapping intents
- deeper semantic understanding

So it should be treated as a baseline, not a final classifier.

## Current Status Summary

What exists now:

- shared core schemas and utilities
- a rule-based intent classifier
- a manual intent classification CLI
- a reusable evaluation pipeline
- a minimal document-loading module
- a heuristic document classification layer
- a dataset adapter layer for benchmark/source datasets such as UltraDomain
- a minimal chunking module
- a modular retrieval layer with keyword and BM25 retrievers
- a minimal end-to-end chatbot pipeline
- a modular LLM client layer
- a first RAG answer-generation layer
- a retrieval evaluation path for UltraDomain benchmark questions
- a retrieval-planning experiment layer for comparing naive vs metadata-aware vs intent-aware retrieval
- automated tests for classification and evaluation
- automated tests for ingestion, chunking, retrieval, pipeline assembly, and answer generation
- a synthetic intent dataset with train/dev/test splits

What does not exist yet:

- reranking
- vector database integration
- API
- UI
- answer-quality evaluation

So the repository is currently focused on the first experimental layer of a larger modular RAG system.
