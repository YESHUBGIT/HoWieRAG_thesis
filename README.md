# howie-rag
This project will be built step by step as a modular RAG pipeline.

## Step 1: Intent classification

Current intent labels:

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

Example commands:

```bash
python scripts/classify_intent.py "What are the key findings?"
python scripts/classify_intent.py "Compare study A and study B"
python scripts/classify_intent.py "What do these findings imply for policy?"
python scripts/evaluate_rule_based_intent.py
python scripts/evaluate_rule_based_intent.py intent_dataset/v3/howie_intent_dataset_390_13intents_v3.csv test
```

## Step 2: Minimal document loading

The next module loads `.txt`, `.md`, and `.pdf` files into `Document` objects so later stages can chunk and retrieve them.

The loader also applies a first heuristic document classification and stores metadata such as:

- `document_type` (`narrative`, `statistical`, `mixed`)
- `has_tables`
- `has_figures`

## Dataset adapters

The project also supports dataset-style adapters in addition to normal file ingestion.

The first adapter is for UltraDomain JSONL files, which can be used in two ways:

- as a source corpus via `context`
- as a benchmark QA dataset via `input`/`question` and `answers`

Suggested workflow:

```bash
python scripts/prepare_ultradomain.py "UltraDomain Dataset" ultradomain_processed
python scripts/build_ultradomain_index.py ultradomain_processed/documents.jsonl ultradomain_index
python scripts/evaluate_ultradomain_retrieval.py ultradomain_processed/questions.jsonl ultradomain_index/chunks.jsonl bm25 ultradomain_results
```

## Step 11: Retrieval planning experiments

The project now supports retrieval-planning experiments over UltraDomain to compare:

- `naive`
- `document_aware`
- `intent_document_aware`

These experiments reuse the existing retrievers (`keyword`, `bm25`) and the existing rule-based intent classifier.

Example command:

```bash
python scripts/evaluate_ultradomain_retrieval_planning.py \
  --documents ultradomain_processed/documents.jsonl \
  --questions ultradomain_processed/questions.jsonl \
  --retriever bm25 \
  --variant all \
  --top-k 5 \
  --candidate-pool-size 30
```

## Step 3: Minimal chunking

The next module splits `Document` objects into smaller `Chunk` objects using a simple fixed-size text chunker with configurable overlap.

## Step 4: Minimal retrieval

The next module retrieves the most relevant chunks for a query using a simple keyword-overlap score.

The retrieval layer is now modular and currently supports:

- `keyword`
- `bm25`

## Step 5: Minimal chatbot flow

The next module connects intent classification, document loading, chunking, and retrieval into one simple end-to-end chatbot pipeline.

Example command:

```bash
python scripts/run_simple_chatbot.py "What trend do we see in student mobility?" path/to/documents
python scripts/run_simple_chatbot.py "What trend do we see in student mobility?" path/to/documents bm25
```

## Step 6: LLM-backed RAG answering

The next module sends the retrieved context to a local vLLM server and generates a grounded answer.

Environment variables:

- `HOWIE_LLM_BASE_URL` defaults to `http://localhost:8000`
- `HOWIE_LLM_MODEL` defaults to `Qwen/Qwen2.5-14B-Instruct`

Example command:

```bash
python scripts/run_rag_chatbot.py "What trend do we see in student mobility?" path/to/documents
python scripts/run_rag_chatbot.py "What trend do we see in student mobility?" path/to/documents bm25
```
