# HoWiE Intent Classification Dataset v3

This package contains a synthetic starter dataset for evaluating intent classification in the HoWiE RAG chatbot prototype.

HoWiE is intended as a retrieval-augmented chatbot for user-centered exploration of higher-education and science-research study results. The dataset focuses on classifying the user question before retrieval and answer generation.

## Files

- `howie_intent_dataset_390_13intents_v3.csv`
- `howie_intent_dataset_390_13intents_v3.jsonl`
- `howie_intent_dataset_390_13intents_v3_README.md`

## Size

- Total examples: 390
- Intents: 13
- Examples per intent: 30
- Split: 20 train / 5 dev / 5 test per intent

## Intents

1. `FACT` — asks for a specific factual detail.
2. `SUMMARY` — asks for an overview, abstract, or key findings.
3. `COMPARISON` — compares studies, methods, groups, results, countries, or time periods.
4. `METHOD_CONTEXT` — asks about methodology, participants, study design, dataset context, variables, or sampling.
5. `LIMITATION` — asks about caveats, bias, uncertainty, weaknesses, or restricted validity.
6. `TREND_PATTERN` — asks about trends, trajectories, changes over time, distributions, or recurring patterns.
7. `EXPLANATION` — asks why/how something happened or what a mechanism means.
8. `SOURCE_SEEKING` — asks for provenance, citation, report, paper, table, page, or evidence source.
9. `NAVIGATION` — asks where in the corpus, interface, portal, or document structure to look.
10. `INTERPRETATION` — asks for analytical meaning, implications, or reasoned reading of findings.
11. `DECISION_SUPPORT` — asks what action to take, what option is preferable, or which intervention seems most useful.
12. `FOLLOWUP` — depends on previous conversational context.
13. `UNKNOWN` — fallback for unrelated, unsupported, or non-HoWiE requests.

## Important evaluation note about FOLLOWUP

`FOLLOWUP` is intentionally different from most other labels. It is partly an intent and partly a conversational-state flag because examples like “What about rural students?” cannot be fully interpreted without previous dialogue context.

For single-turn ML or LLM experiments, explicitly document that `FOLLOWUP` is a harder target than the semantic labels. A low score on `FOLLOWUP` does not necessarily mean the model has poor semantic understanding; it may mean the setup lacks conversation history. In later versions, you may model this as a separate boolean feature such as `requires_context` instead of, or in addition to, an intent label.

## Columns

- `id`: stable example identifier.
- `question`: user question.
- `intent`: gold intent label.
- `split`: `train`, `dev`, or `test`.
- `language`: currently `en`.
- `domain`: currently `higher_education_research` for in-domain examples or `out_of_scope` for clearly unrelated UNKNOWN examples.
- `source_type_hint`: rough source type likely needed by a later RAG system. UNKNOWN rows use `none`.
- `requires_context`: whether the question depends on previous conversation context. In CSV this is stored as `true`/`false`; in JSONL it is stored as real JSON booleans.

## v3 changes

Compared with v2:

1. Clearly unrelated `UNKNOWN` examples now use `domain=out_of_scope`.
2. `UNKNOWN` examples now use `source_type_hint=none`.
3. The README now explicitly warns that `FOLLOWUP` is a harder and somewhat different target in single-turn evaluation.
4. The JSONL file keeps `requires_context` as real JSON booleans, not strings.

## Experimental use

Recommended setup:

- Rule-based classifier: use no training data. Tune only on development data if needed.
- ML classifier: train on `train`, tune on `dev`, report final results on `test`.
- LLM classifier: evaluate zero-shot or few-shot on the same `test` split. Keep prompts fixed after development.

Recommended metrics:

- Accuracy
- Macro F1
- Per-class precision / recall / F1
- Confusion matrix
- Latency
- Cost per query for LLM-based classification
- Failure examples by intent

## Limitations

This is a fully synthetic, perfectly balanced starter dataset. It is useful for controlled baseline comparison and for testing your experiment pipeline, but it is not sufficient for strong claims about production readiness or real-world generalization.

Expected limitations:

- Synthetic phrasing may be cleaner than real user questions.
- Class balance may produce optimistic metrics.
- Real users will ask ambiguous, mixed-intent, incomplete, multilingual, and poorly phrased questions.
- SOURCE_SEEKING and NAVIGATION may be confused and should be reviewed using the confusion matrix.
- FOLLOWUP requires conversation history and should be interpreted carefully in single-turn evaluation.

## Recommended next improvement

After this starter dataset works, add a small human-reviewed hard set with ambiguous and natural user phrasing. A good first hard set would contain 5–10 manually reviewed examples per intent, including mixed cases such as:

- source-seeking vs navigation
- explanation vs interpretation
- trend-pattern vs comparison
- limitation vs method context
- follow-up questions with and without previous-turn context

Do not use the hard set for training. Use it as an additional stress-test after reporting the clean test-set results.

## Counts

### By split

{'train': 260, 'dev': 65, 'test': 65}

### By domain

{'higher_education_research': 360, 'out_of_scope': 30}

### By intent

{'FACT': 30, 'SUMMARY': 30, 'COMPARISON': 30, 'METHOD_CONTEXT': 30, 'LIMITATION': 30, 'TREND_PATTERN': 30, 'EXPLANATION': 30, 'SOURCE_SEEKING': 30, 'NAVIGATION': 30, 'INTERPRETATION': 30, 'DECISION_SUPPORT': 30, 'FOLLOWUP': 30, 'UNKNOWN': 30}
