# Thesis Notes

Canonical thesis-facing notes: current findings, conclusions, algorithms,
pipelines, and references to the code. These notes should stay conceptual and
should not contain sbatch commands, scratch launch recipes, or debugging logs.

## Naming convention

```
NNN-short-kebab-slug.md
```

- `NNN` — zero-padded, monotonically increasing index (`001`, `002`, …). Never
  reuse or renumber; the number is a stable handle to cite from other notes.
- `short-kebab-slug` — 3–6 word topic.

Each note starts with a metadata block: date, status, the runs/configs it draws
on, and links to related notes (`[[002-...]]`) and source docs/scripts.

When a note supersedes or refines another, say so in both (forward + back
reference) rather than editing the old one into silence.

## Index

| # | Note | Topic |
|---|------|-------|
| 001 | [001-model-scale-coverage-vs-reliability.md](001-model-scale-coverage-vs-reliability.md) | Model scale vs. pass@k: the "smaller-is-broader" coverage effect is a judge artifact (specificity-preserving support → monotonic bigger-is-better) |
| 002 | [002-taxonomy-mapping-and-rollout-metrics.md](002-taxonomy-mapping-and-rollout-metrics.md) | Taxonomy-aware metric construction: paper-aligned text→node mapping (`cascade` = CLIP-t2t + `vlm-eval` thresholds; direct measures over rollout texts) and rollout aggregation — binary `pass@k` from judge verdicts + graded hF from deduped per-rollout metrics for later statistics |
| 003 | [003-recursive-self-aggregation.md](003-recursive-self-aggregation.md) | Recursive Self-Aggregation (RSA, arXiv:2509.26626): the test-time population-aggregation method + aggregation-aware RL recipe; our OVEN test-time adaptation (`run_recursive_self_agg.py`); planned aggregation-aware RL for taxonomy-aware image classification |
| 004 | [004-taxonomy-aware-parametric-traversal-grpo.md](004-taxonomy-aware-parametric-traversal-grpo.md) | Taxonomy-Aware Parametric Traversal GRPO: reframing beyond "RSA+OVEN" into structured visual-semantic search; unlockable-knowledge mining; traversal prompt design; shaped reward with conservative linker + specific_hF + path_match; six-experiment matrix (launched, results pending) |

Operational commands for these notes belong in `../commands.md` or
`../operations/`.
