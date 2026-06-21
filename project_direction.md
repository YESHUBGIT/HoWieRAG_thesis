# HoWie-RAG Project Direction

## Purpose of This Document

This document is the running reference for the current state, rationale, discoveries, and likely research direction of the `howie-rag` project.

It is meant to capture:

- what has already been built
- what we learned from building and testing it
- what literature suggests about the novelty of the project
- what the likely thesis direction should be
- what the most useful next steps are

This should be updated as the repository grows.

## Project Goal

`howie-rag` is a modular RAG chatbot research prototype for exploring higher-education and science-research documents.

The project is not only about answering questions. The intended long-term goal is to help users:

- find relevant research evidence
- summarize findings
- compare studies or reports
- inspect methods and limitations
- locate sources and supporting documents
- receive grounded, evidence-aware answers from a local RAG system

The project is being built step by step. The strategy has been:

1. first make a minimal end-to-end system work
2. then identify weak modules
3. then improve and evaluate those modules experimentally

## Why This Direction Was Chosen

Instead of starting with advanced methods immediately, the project first built a simple but complete baseline pipeline. This was important because it allowed us to see real bottlenecks rather than guessing them in advance.

The current philosophy is:

- build a working baseline first
- keep modules small and testable
- improve the system after integration exposes real failure modes

## What Has Been Built So Far

### 1. Core schemas and utilities

The project includes basic schemas for:

- `IntentResult`
- `Document`
- `Chunk`

It also includes `stable_id(text: str) -> str`, which creates short deterministic IDs from text.

### 2. Intent classification module

We built a rule-based intent classifier with these labels:

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

The current classifier is keyword- and priority-based.

This is the first query-understanding layer of the system.

### 3. Intent evaluation pipeline

We built a reusable evaluation module for intent classification.

It computes:

- accuracy
- macro precision / recall / F1
- per-label metrics
- confusion matrix

This allows direct comparison between future intent classifiers, such as:

- rule-based
- classical ML
- LLM-based

### 4. Intent dataset

We created and refined a 13-intent dataset.

The best current version is `intent_dataset/v3`.

Properties:

- 390 examples
- 13 intents
- balanced split
- 20 train / 5 dev / 5 test per intent
- explicit `requires_context` flag for follow-up questions

This dataset is synthetic and balanced. It is a useful benchmark and starter experiment set, but not enough for strong real-world generalization claims.

### 5. Ingestion

We built a minimal ingestion module that loads files from disk into `Document` objects.

It currently supports:

- `.txt`
- `.md`
- `.pdf`

### 6. PDF processing

We added minimal PDF support with `pypdf`.

The current PDF ingestion:

- extracts text page by page
- inserts markers like `[Page 1]`
- stores extracted content as normal document text

This is enough for first-pass PDF use, but it is still basic and may be weak on:

- scanned PDFs
- multi-column layouts
- complex tables
- structure-heavy codebooks

### 7. Chunking

We built a minimal chunking module.

Current method:

- fixed-size character chunking
- configurable overlap
- returns `Chunk` objects with metadata

This is a baseline chunker and not yet document- or section-aware.

### 8. Retrieval

We built a minimal lexical retriever.

Current method:

- tokenize query and chunk text
- score by token overlap
- remove stopwords
- return top matching chunks

This is the current baseline retriever.

### 9. End-to-end non-LLM pipeline

We connected:

- intent classification
- ingestion
- chunking
- retrieval

This gave us the first working end-to-end retrieval pipeline.

### 10. Local LLM integration

We added a modular LLM layer and connected the app to a local vLLM server.

Current working local model:

- `Qwen/Qwen2.5-14B-Instruct`

The integration uses:

- an abstract `BaseLLMClient`
- a `VLLMClient`
- a RAG answer-generation layer that builds prompts from question, intent, and retrieved chunks

### 11. End-to-end LLM-backed RAG chatbot

The current full path is:

1. user asks a question
2. classify intent
3. load documents
4. extract PDF text if needed
5. chunk documents
6. retrieve relevant chunks
7. build grounded prompt
8. query local vLLM server
9. return answer

This means a full minimal local RAG chatbot now exists.

### 12. Tests and documentation

The repository includes tests for:

- setup
- intent classification
- intent evaluation
- ingestion
- chunking
- retrieval
- chatbot pipeline assembly
- RAG answer-generation prompt logic

It also includes repo documentation in:

- `README.md`
- `file_structure_detail.md`
- this file: `project_direction.md`

### 13. UltraDomain retrieval-planning experiment layer

The project now includes the first explicit retrieval-planning experiment layer for UltraDomain.

Three retrieval variants can now be compared:

- `naive`
- `document_aware`
- `intent_document_aware`

This layer reuses:

- the existing modular retrievers (`keyword`, `bm25`)
- the existing rule-based intent classifier
- the existing document classification metadata

The current implementation works through score boosting and reranking, not hard filtering.

This is important because it is the first concrete implementation of the thesis idea that retrieval should depend on both question type and source/document characteristics.

## What We Learned From Testing the System

### 1. The basic pipeline works

We successfully confirmed that:

- intent classification works
- ingestion works
- PDF extraction works
- retrieval works
- the local LLM can answer from retrieved context

So the project is beyond isolated components. It is a functioning baseline system.

### 2. Retrieval is the main current bottleneck

When we tested the pipeline, many weak results were not caused by the LLM alone. They were caused by weak retrieval.

Typical failure mode:

- the query is vague or weakly matched
- the retriever brings in partially relevant chunks
- the LLM produces a plausible but weakly grounded answer

This means the next technical bottleneck is retrieval quality.

### 3. Chunking also matters a lot

The fixed-size chunker can split documents awkwardly, especially in PDFs.

This affects:

- retrieval precision
- readability of retrieved evidence
- final answer quality

### 4. Corpus quality strongly affects the system

The chatbot can only answer well if the source corpus actually contains the right kind of information.

Early source data was too noisy and random, which led to poor chatbot behavior.

This forced the realization that source suitability is a major part of system quality.

### 5. Different source corpora support different intents differently

The current DZHW PDF corpus is especially strong for:

- `FACT`
- `METHOD_CONTEXT`
- `NAVIGATION`
- `SOURCE_SEEKING`
- `LIMITATION`

It is weaker for:

- `TREND_PATTERN`
- `INTERPRETATION`
- `DECISION_SUPPORT`

because the current documents are more documentation-heavy than findings-heavy.

### 6. Conservative refusal behavior is good

For out-of-domain or unsupported questions, the system sometimes correctly refuses or says the evidence is insufficient.

This is a positive sign. A grounded refusal is better than a fluent hallucination.

## Current Data Situation

The active `source_data/` folder now contains DZHW-related source files, including:

- data package overviews
- methods reports
- codebooks
- variable lists
- release notes
- interview guideline PDFs

These are domain-matched and much better than the earlier noisy test corpus.

### Current strengths of this corpus

- strong for methodological and documentation questions
- strong for source and navigation questions
- realistic academic and institutional document style
- useful for testing PDF ingestion and document-heavy RAG

### Current limitations of this corpus

- less suitable for broad findings/trends unless those are explicitly discussed in the reports
- may produce weak answers for questions that ask for analytical or comparative findings not contained in the documents
- codebooks and variable lists may be difficult for simple PDF extraction and naive chunking

## Source and Data Type Taxonomy

One major design discovery is that not all sources should be treated the same way.

Different source types support different user intents, and this should influence:

- ingestion
- chunking
- retrieval
- answer generation
- evaluation

The project now has enough variety in its sources that source typing is necessary rather than optional.

### Main source/data types in HoWie-RAG

#### 1. Narrative report text

Examples:

- research reports
- executive summaries
- project overviews
- policy reports
- findings sections
- explanatory web pages

Characteristics:

- paragraph-based
- explanatory
- well suited to standard chunk-and-retrieve RAG

Best for:

- `SUMMARY`
- `EXPLANATION`
- `INTERPRETATION`
- `LIMITATION`
- `DECISION_SUPPORT`

Handling:

- heading-aware chunking
- paragraph or section chunking
- standard lexical or semantic retrieval

#### 2. Method / methodology documents

Examples:

- data and methods reports
- methodological notes
- survey design reports
- technical background reports

Characteristics:

- explanatory but technical
- rich in definitions, scope, caveats, and procedures

Best for:

- `METHOD_CONTEXT`
- `LIMITATION`
- `SOURCE_SEEKING`
- `NAVIGATION`

Handling:

- chunk by section or subsection
- preserve headings carefully
- prioritize method-like terminology in retrieval later

#### 3. Codebooks / variable documentation

Examples:

- codebooks
- variable lists
- field dictionaries
- data dictionaries

Characteristics:

- semi-structured
- variable-centric
- often list- or table-like rather than narrative

Best for:

- `FACT`
- `METHOD_CONTEXT`
- `SOURCE_SEEKING`
- `NAVIGATION`

Handling:

- not ideal for naive paragraph chunking
- better as entry-based or variable-based chunks
- variable names and labels should be preserved exactly

#### 4. Tabular statistical data

Examples:

- CSV tables
- XLS tables
- HTML tables
- PDF table exports

Characteristics:

- highly numeric
- row/column semantics matter
- weak as flat text-only narrative input

Best for:

- `FACT`
- `TREND_PATTERN`
- `COMPARISON`

Handling:

- should not be treated like normal prose
- better as row-wise, row-block, or table-aware chunks
- row labels, years, units, and footnotes should be preserved
- later, rows may be transformed into normalized text statements for easier retrieval

#### 5. Graphics / chart pages

Examples:

- graph HTML pages
- visual dashboard pages
- chart metadata pages

Characteristics:

- often limited explanatory prose
- visual meaning may be partially lost in plain text extraction

Best for:

- `NAVIGATION`
- `SOURCE_SEEKING`
- some `TREND_PATTERN` if captions are useful

Handling:

- preserve title, caption, labels, and source note where possible
- lower priority for direct answer generation unless text is rich enough

#### 6. Metadata / catalog / landing pages

Examples:

- GovData metadata pages
- publication catalogue entries
- dataset overview pages
- source directory pages

Characteristics:

- descriptive rather than substantive
- useful for finding sources, less useful for deep answers

Best for:

- `SOURCE_SEEKING`
- `NAVIGATION`

Handling:

- store as metadata-rich source objects
- do not rely on them as the main answer evidence for fact-heavy questions

#### 7. Release notes / version notes

Examples:

- release notes
- changelogs
- version histories

Characteristics:

- narrow scope
- useful mainly for provenance and change tracking

Best for:

- `SOURCE_SEEKING`
- `NAVIGATION`
- some `LIMITATION`

Handling:

- usually small chunks or whole-file chunks

#### 8. Instruments / interview guidelines / questionnaires

Examples:

- interview guidelines
- survey instruments
- questionnaires

Characteristics:

- structured prompts/questions
- method-heavy, not findings-heavy

Best for:

- `METHOD_CONTEXT`
- `LIMITATION`
- `SOURCE_SEEKING`

Handling:

- preserve section or item boundaries
- chunk by section or block rather than pure fixed size

### Recommended working taxonomy in the repo

The best concrete source-type taxonomy for HoWie-RAG at the moment is:

- `narrative_report`
- `method_report`
- `codebook`
- `variable_list`
- `statistical_table`
- `graphic_page`
- `metadata_page`
- `release_notes`
- `instrument`

### Why this matters for downstream modules

The major design implication is:

- narrative sources should be handled differently from structured or tabular sources

In particular:

- narrative sources -> paragraph or section-aware chunking and standard RAG retrieval
- method sources -> method-focused chunking and retrieval
- codebook and variable sources -> entry-based or variable-based chunking
- statistical tables -> row-wise or table-aware handling
- metadata pages -> navigation and provenance support

### Impact on retrieval

Source types should influence which documents are preferred for which intents.

Examples:

- `SUMMARY` -> prefer `narrative_report`
- `METHOD_CONTEXT` -> prefer `method_report`, `instrument`, `codebook`
- `SOURCE_SEEKING` -> prefer `metadata_page`, `method_report`, overview sources
- `FACT` -> prefer `statistical_table`, `codebook`, `variable_list`
- `TREND_PATTERN` -> prefer `statistical_table`
- `LIMITATION` -> prefer `method_report`

This is one of the strongest paths toward question-type-aware retrieval.

### Impact on answer generation

Source type should also influence how the final answer is formed.

- narrative sources support direct natural-language synthesis
- tabular sources may require value extraction or row normalization first
- metadata sources are better for source/navigation answers than for substantive findings

### Current implication for the GovData corpus

The newly harvested GovData/BMFTR sources are especially numerical and tabular.

This means they are valuable, but mainly for:

- `FACT`
- `TREND_PATTERN`
- `COMPARISON`

They are less suitable as-is for:

- `SUMMARY`
- `INTERPRETATION`
- `DECISION_SUPPORT`

unless paired with more narrative report-style sources.

This confirms that GovData should be treated as a separate source family with different downstream handling, not simply merged into the same flow as narrative PDFs.

### Best next source-handling decision

Before improving retrieval too much further, the project should:

1. store source type metadata explicitly
2. classify current corpora into those source types
3. improve chunking based on source type
4. improve retrieval based on both intent and source type

## LLM Serving Status

The working local serving stack is:

- Torch `2.5.1+cu121`
- CUDA `12.1`
- vLLM `0.6.6.post1`
- Transformers `4.46.3`
- Tokenizers `0.20.3`
- model: `Qwen/Qwen2.5-14B-Instruct`

Originally, `Qwen3` was considered, but compatibility issues with the available driver/runtime led to adopting `Qwen2.5-14B-Instruct` as the stable working local model.

This is acceptable and pragmatic because a stable local LLM is more useful than an unstable newer model.

## Why Some Advanced Methods Were Not Used Yet

The following were discussed but intentionally not introduced in the first phase:

- TF-IDF retrieval
- BM25 retrieval
- embedding retrieval
- vector database
- `bge-m3`
- rerankers
- logistic regression intent classifier

Reason:

- the project first needed a complete working baseline
- advanced methods are more useful once the baseline exposes real bottlenecks

Now that the system works, those methods become meaningful future upgrades.

## Literature Review: Main Implications for HoWie-RAG

The `lit_review/` folder contains foundational RAG papers, surveys, evaluation papers, and domain papers.

The key conclusion from reviewing this material is:

### A generic “RAG chatbot for education” is not enough as a thesis contribution

There is already substantial work on:

- general RAG systems
- RAG for education
- RAG evaluation
- adaptive RAG
- retrieval improvements

So if the project is framed only as:

- “I built a RAG chatbot for higher education”

then the research contribution is too generic.

### The project is still on the right track if it sharpens its contribution

The strongest literature-aligned but still distinct direction is:

- question-type-aware retrieval
- evidence-faithful answering
- document-type-aware handling of higher-education research documents
- adaptive retrieval decisions

These align strongly with what is already built in HoWie-RAG.

## Most Relevant Literature Insights

### 1. Original RAG paper

The original RAG work justifies the baseline architecture:

- retrieve supporting documents
- generate answer from retrieved evidence

This corresponds to the system already built.

### 2. RAG surveys

The surveys distinguish between:

- naive RAG
- advanced RAG
- modular RAG

HoWie-RAG currently sits at the naive-to-modular transition:

- already modular
- but still using baseline chunking and retrieval

### 3. `Know your RAG`

This paper is especially important.

Its key idea:

- retrieval performance depends on question/context label type
- evaluation datasets should reflect actual user question types

This strongly validates HoWie-RAG’s intent taxonomy and suggests a strong research path:

- use question types to drive retrieval strategies and evaluation

### 4. Adaptive RAG for conversational systems

This paper argues that retrieval is not always needed for every conversational turn.

This is highly relevant for HoWie-RAG because it already has:

- `FOLLOWUP`
- `UNKNOWN`
- multiple question types

This suggests a future retrieval gate:

- retrieve only when useful
- otherwise clarify, refuse, or answer differently

### 5. Correctness is not faithfulness

This is very important for HoWie-RAG’s domain.

The key lesson:

- a correct-looking answer is not enough
- the answer must actually be supported by cited evidence

For a research-document chatbot, this is a highly relevant gap.

### 6. RAG in education survey

This survey shows that there are already many RAG chatbots in education.

This means HoWie-RAG must differentiate itself not by merely being “an education chatbot”, but by a more specific contribution.

## Current Thesis-Relevant Direction

The best current thesis direction is not:

- “A RAG chatbot for education”

The better direction is something like:

### Working contribution framing

**Question-Type-Aware and Evidence-Faithful Retrieval-Augmented Generation for Higher-Education Research Documents**

or equivalently:

**A modular local RAG system for higher-education research exploration, improved with question-type-aware retrieval and evidence-faithful answering**

## Why This Direction Is Strong

Because HoWie-RAG already has the necessary ingredients:

- explicit question types
- a modular pipeline
- PDF-heavy academic documents
- local LLM answering
- observable retrieval bottlenecks
- a domain where trust, method context, and source-faithfulness matter

This means the system is already well positioned to go beyond generic RAG.

## Best Next Research Opportunities

### 1. Question-type-aware retrieval

This is currently the strongest next step.

Idea:

- different intents trigger different retrieval strategies

Examples:

- `FACT` -> precise chunk retrieval
- `SUMMARY` -> broader evidence set
- `METHOD_CONTEXT` -> prioritize methods/codebook-like sources
- `SOURCE_SEEKING` -> prioritize metadata/source documents
- `LIMITATION` -> prioritize methods or limitation-heavy sections

This is well aligned with both the current pipeline and the literature.

### 2. Adaptive retrieve-or-not gate

Another strong direction.

Idea:

- not every question should automatically trigger retrieval
- some questions should instead:
  - ask for clarification
  - refuse
  - answer from available context

This is especially relevant for vague questions and follow-ups.

### 3. Evidence-faithful answering

This is highly suitable for research-document chatbots.

Idea:

- answers should be explicitly tied to supporting chunks
- unsupported claims should be avoided or marked as uncertain
- source references should be meaningful, not decorative

### 4. Document-type-aware retrieval

This is very practical for the current corpus.

Idea:

- methods reports, overviews, codebooks, release notes, and variable lists should not all be treated identically

Examples:

- `METHOD_CONTEXT` -> methods report / instrument first
- `FACT` -> variable list / codebook / overview
- `SOURCE_SEEKING` -> overview / methods report / file metadata

## What Should Probably Happen Next Technically

The most useful next technical work is:

1. improve retrieval
2. improve chunking
3. improve corpus quality/coverage
4. then evaluate and compare advanced methods

### Retrieval improvement candidates

- TF-IDF
- BM25
- embedding retrieval with `bge-m3`
- hybrid retrieval
- reranking

## Structure-Aware Document and Data Classification Direction

Another major design direction is to introduce a document/data classifier before chunking and embedding.

This is becoming important because many real sources are not homogeneous documents. A single PDF may contain:

- headings
- narrative paragraphs
- statistical tables
- figures or charts
- captions
- metadata-like sections
- appendices

So the system should not treat the whole PDF as plain text only.

### Core idea

The system should classify:

1. the whole document type
2. the internal content blocks inside the document

This would allow chunking, retrieval, and later indexing to become structure-aware.

### Whole-document type classification

At the document level, the classifier should identify broad document types such as:

- `narrative`
- `statistical`
- `graphical`
- `mixed`

This document-level classification would help decide the initial processing strategy.

### Internal block classification

At the block level, the system should identify content units such as:

- `heading`
- `paragraph`
- `table`
- `graph_or_figure`
- `caption`
- `metadata`

This is important because many files are mixed documents, especially PDFs.

### Why this is needed

Most PDFs in the real world contain multiple content forms, not a single clean narrative flow. If those are flattened into plain text too early, the system loses important structure.

That harms:

- chunk quality
- retrieval precision
- evidence faithfulness
- support for source-seeking questions

### Better target pipeline

The long-term improved pipeline should look like:

1. file upload / document ingestion
2. layout-aware parsing
3. document-level classification
4. block-level classification
5. type-specific chunking
6. separate storage and indexing
7. query-type-based retrieval
8. grounded answer generation with citations

### Recommended chunking strategies by content type

#### Narrative text

- chunk by section, heading, and paragraph

#### Tables

- preserve table structure
- store structured rows where possible
- create table summaries for later text-based retrieval

#### Graphs / figures

- preserve figure reference
- preserve caption
- preserve nearby explanatory text
- mark clearly that the system is not yet performing true visual interpretation of the figure itself

#### Mixed documents

- keep all block types separately
- preserve links between them using page number, section, and document ID

### Recommended storage/indexing families

The system should eventually store different output types separately, for example:

- `narrative_chunks`
- `table_summaries`
- `structured_tables`
- `figure_captions`
- `document_summaries`
- `metadata`

This separation makes later retrieval and evaluation cleaner.

### Why this is a thesis-strength contribution

This makes HoWie-RAG more intelligent than normal flat-text RAG because it understands:

- what kind of document it is reading
- what kind of content block it is using
- how that block should be chunked
- how that block should be indexed
- how that block should be retrieved for different question types
- what the system can and cannot interpret reliably

This is a strong and defensible contribution because it is directly tied to realistic higher-education research documents, which are often mixed-format and evidence-sensitive.

### Practical implementation order

This direction should be implemented in stages.

#### Stage 1

Add a simple document-level classifier:

- `narrative`
- `statistical`
- `graphical`
- `mixed`

#### Stage 2

Add simple block-level classification:

- `heading`
- `paragraph`
- `table`
- `caption`
- `metadata`

#### Stage 3

Add type-specific chunking strategies.

#### Stage 4

Store chunks and extracted units separately by content type.

#### Stage 5

Use both question intent and source/block type in retrieval.

### Chunking improvement candidates

- paragraph-aware chunking
- section-aware chunking
- page-aware chunking for PDFs
- document-type-aware chunking

### Intent classification experimental candidates

- TF-IDF + logistic regression
- local LLM classifier
- later transformer-based classifier if needed

## Where Different Methods Fit in the Pipeline

### Intent classification stage

Possible methods:

- rule-based
- logistic regression
- transformer classifier
- LLM classifier

### Retrieval stage

Possible methods:

- token overlap
- TF-IDF
- BM25
- dense retrieval with `bge-m3`
- vector search / vector DB
- hybrid retrieval
- reranking

### Answer generation stage

Possible future improvements:

- better prompt design
- retrieval confidence checks
- answer faithfulness checks
- source-aware answer formatting

## Position on Document Processing Tools

We discussed several alternatives for document extraction and conversion.

### Current position

The custom pipeline remains useful because it is:

- understandable
- controllable
- easy to test
- modular

### Future possibility

Tools such as:

- Marker
- PyMuPDF
- Docling
- markitdown
- LlamaParse

may later be introduced as optional ingestion backends.

At the moment, the correct approach is to compare them empirically on the real corpus, not assume one is automatically best.

## Current Project Status in One Sentence

HoWie-RAG is now a working local modular RAG baseline over higher-education research documents, with intent classification, PDF ingestion, chunking, retrieval, and local LLM answering, and the strongest emerging research direction is to improve it using question-type-aware retrieval and evidence-faithful answering.

## Most Important Current Decision

The project should no longer be framed simply as:

- “a RAG chatbot for education”

It should be framed as:

- a modular local RAG system for higher-education research exploration
- with a contribution in retrieval adaptation, evidence faithfulness, and domain/document-type-aware behavior

## Suggested Near-Term Action Plan

1. Continue using the current system as the baseline.
2. Improve retrieval before changing too many other parts.
3. Make chunking more PDF- and document-aware.
4. Evaluate the system with corpus-appropriate questions.
5. Design the next experiments around:
   - question-type-aware retrieval
   - retrieval gating
   - answer faithfulness

## Working Thesis Direction

If a concise thesis direction is needed right now, the best current phrasing is:

> Start from a standard local RAG chatbot for higher-education research documents, then improve it with question-type-aware retrieval, adaptive retrieval decisions, and evidence-faithful answering.

This is the most defensible direction based on what has already been built and what the literature suggests.
