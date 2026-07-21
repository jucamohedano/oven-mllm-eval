# Model Diversity, Judge Deduplication, and Coverage

This note keeps the useful diagnostic content from the earlier dedup and key
findings notes, but updates the interpretation to match the later audit.

## Dedup observation

Judge runs deduplicate identical prompts before sending them to the judge model.
The dedup ratio is therefore an indirect signal of rollout diversity: if many
rollouts repeat the same text, the judge workload shrinks; if rollouts are more
diverse, more unique prompts remain.

Earlier runs showed that smaller models could produce broader, more varied
candidate sets at high sampling counts. This helped explain why large-k pass@k
sometimes looked better for smaller models.

## Updated interpretation

The later false-positive audit changed the conclusion. The apparent large
"2B beats 8B at high k" story was mostly judge inflation caused by accepting
under-specific or otherwise weak positives. When positives are restricted to a
specificity-preserving support set, the 2B advantage becomes much smaller or
disappears depending on the metric.

The current thesis framing is:

- high-k pass@k measures coverage of judge-accepted rollouts;
- diversity can help coverage, but it can also create more opportunities for
  judge false positives;
- larger models tend to have stronger pass@1/reliability, while smaller models
  may appear broader under lenient acceptance;
- supported coverage and specificity-aware taxonomy metrics are needed to
  separate real coverage from lenient judging.

## Diagnostics to keep

Useful diagnostics for future runs:

- unique prompt count before judging;
- dedup savings per model;
- judge-positive rollouts per hit example;
- supported-positive rollouts per hit example;
- pass@1 versus pass@full;
- mapped coverage and all-example taxonomy scores.

