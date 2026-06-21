# HoWiE Intent Classification Dataset — 13 Intents, v2

This is a synthetic starter dataset for comparing intent classifiers in the HoWiE RAG chatbot prototype. It is designed for controlled baseline experiments, not for production-readiness claims.

## v2 fixes

1. Corrected `HOWIE-FAC-024`: the old question, `Which table contains the employment rate of graduates?`, was provenance/navigation-like but labeled as `FACT`. It has been replaced with the factual question `What was the employment rate of graduates?`.
2. JSONL now stores `requires_context` as real JSON booleans (`true` / `false`), not strings (`"true"` / `"false"`).
3. README now explicitly warns that the dataset is synthetic and balanced, so baseline results may be optimistic and should not be used as evidence of real-world production performance.

## Dataset size

- Total examples: 390
- Intents: 13
- Examples per intent: 30
- Splits: train=260, dev=65, test=65

Each intent has exactly:

- 20 train examples
- 5 dev examples
- 5 test examples

## Intent labels

- `FACT` — asks for a specific factual detail, number, definition, value, or attribute.
- `SUMMARY` — asks for an overview, short synthesis, or key findings.
- `COMPARISON` — asks to compare studies, groups, methods, countries, results, or time periods.
- `METHOD_CONTEXT` — asks about methodology, participants, design, sample construction, data collection, or dataset context.
- `LIMITATION` — asks about caveats, weaknesses, uncertainty, bias, validity, or missing evidence.
- `TREND_PATTERN` — asks about trends, trajectories, changes over time, recurring patterns, or developments.
- `EXPLANATION` — asks why or how something happened, or asks for causal/contextual explanation.
- `SOURCE_SEEKING` — asks for the source, citation, paper, report, page, table, or evidence behind an answer.
- `NAVIGATION` — asks where to look in the corpus, system, portal, section, dashboard, or document collection.
- `INTERPRETATION` — asks what findings mean, imply, suggest, or how they should be analytically read.
- `DECISION_SUPPORT` — asks what action, option, intervention, or policy is preferable based on evidence.
- `FOLLOWUP` — depends on previous conversation context and should often be treated as both an intent and a conversational-state flag.
- `UNKNOWN` — fallback for greetings, unrelated questions, malformed input, or requests outside the system scope.

## Columns

- `id`: unique example identifier
- `question`: user question
- `intent`: gold intent label
- `split`: train/dev/test
- `language`: currently `en`
- `domain`: currently `higher_education_research` or `out_of_scope`
- `source_type_hint`: rough expected source type, useful for later retrieval-routing experiments
- `requires_context`: whether the question depends on previous turns. In CSV this is stored as text; in JSONL this is stored as a true JSON boolean.

## Recommended experimental use

Use the same held-out test split for all classifiers:

1. `RuleBasedIntentClassifier`: no training data; tune only on dev if needed.
2. `MLIntentClassifier`: train on train, tune on dev, final report on test.
3. `LLMIntentClassifier`: use zero-shot or few-shot prompts; tune prompt on dev, final report on test.

Report at minimum:

- Accuracy
- Macro F1
- Per-class precision / recall / F1
- Confusion matrix
- Latency
- Cost per query for LLM-based classification
- Failure examples by intent

## Important limitation

This dataset is fully synthetic and perfectly balanced. That makes it useful for a clean first experiment, but it will likely produce optimistic scores compared with real user data. For a thesis, treat this as a baseline/starter dataset. Stronger claims require expansion with realistic questions from stakeholder workshops, user studies, logs, or manually annotated domain examples.
