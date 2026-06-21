# HoWiE Intent Classification Dataset — 13 Intents

Synthetic starter dataset for comparing intent classifiers in a HoWiE-style RAG chatbot for higher-education and science-research result exploration.

## Size

- Total examples: 390
- Intents: 13
- Examples per intent: 30
- Split per intent: 20 train, 5 dev, 5 test

## Intents

1. `FACT` — asks for a specific factual detail, statistic, date, number, indicator, or definition.
2. `SUMMARY` — asks for an overview, synthesis, key findings, or short summary.
3. `COMPARISON` — asks to compare studies, groups, countries, methods, results, or time points.
4. `METHOD_CONTEXT` — asks about methodology, participants, design, dataset context, indicators, sampling, or measurement.
5. `LIMITATION` — asks about caveats, uncertainty, bias, weaknesses, boundaries, or reliability.
6. `TREND_PATTERN` — asks about trends, trajectories, changes over time, patterns, or developments.
7. `EXPLANATION` — asks why or how something happened, or what factors/mechanisms explain a result.
8. `SOURCE_SEEKING` — asks where an answer, claim, statistic, citation, page, paper, table, or source came from.
9. `NAVIGATION` — asks where in the corpus, system, portal, website, folder, section, or report to look.
10. `INTERPRETATION` — asks for analytical meaning, implications, reasoned reading, or policy relevance.
11. `DECISION_SUPPORT` — asks what action to take, what option is preferable, or what recommendation follows from evidence.
12. `FOLLOWUP` — context-dependent conversational follow-up that cannot be fully interpreted without prior dialogue.
13. `UNKNOWN` — fallback for greetings, unrelated tasks, out-of-domain requests, or unsupported intents.

## Columns

- `id`: stable example identifier
- `question`: user question
- `intent`: gold label
- `split`: train/dev/test
- `language`: currently `en`
- `domain`: currently `higher_education_research`
- `source_type_hint`: broad expected source type
- `requires_context`: `true` for follow-up-like context-dependent examples, otherwise `false`

## Recommended use

- Train classical ML models only on `split == train`.
- Use `dev` for prompt/rule/threshold tuning.
- Use `test` once for final reporting.
- Report accuracy, macro F1, per-class precision/recall/F1, confusion matrix, latency, and LLM cost per query.

## Note on SOURCE_SEEKING vs NAVIGATION

These labels are intentionally separate:

- `SOURCE_SEEKING` asks for evidence provenance: citation, study, page, table, report, DOI, or source document.
- `NAVIGATION` asks where to go in the system/corpus/portal/report to find a section or resource.

If model performance shows persistent confusion between them, you can merge both into a broader `SOURCE_NAVIGATION` label in a secondary experiment.
