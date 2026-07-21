# RSA Runbook

This is the operational companion to `docs/thesis_notes/003-recursive-self-aggregation.md`.
It is intentionally command-oriented; conceptual RSA discussion belongs in the
thesis note.

## Current script

`scripts/run_recursive_self_agg.py` adapts the RSA population-update loop to
OVEN samples.

Two candidate formats are supported:

- `--candidate-format answer` uses an existing `*_samples.jsonl` file and treats
  each row's `all_texts` as the initial population `P1`.
- `--candidate-format solution` first generates concise solution traces ending
  in `\boxed{...}`, then recursively aggregates solution traces.

The output is still compatible with `scripts/run_judge.py` and
`scripts/score_predictions.py`: rows contain `all_texts` and `prediction`, so the
normal judge/scoring pipeline can be reused.

## Operational notes

- Use `--resume` for interrupted runs.
- The default output name is
  `<input>_rsa[_solution]_n{N}_k{K}_t{T}.jsonl`.
- `T` counts population states, so the number of update rounds is `T - 1`.
- `--no-image` is a text-only ablation; otherwise the image is passed at every
  RSA generation/aggregation step.
- Data-parallel sharding writes shard files and merges safely by row id.
- Downstream judge/scoring fields are stripped from source rows before writing
  RSA outputs.

## Scoring RSA outputs

Judge and score RSA outputs exactly like ordinary samples files. Use a distinct
output/summary suffix so baseline and RSA results are not overwritten. Keep the
commands themselves in `docs/commands.md` or experiment-specific notes.

