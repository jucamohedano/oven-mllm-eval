# OVEN MLLM Evaluation Docs

This directory separates thesis-facing notes from operational runbooks and
scratch material.

## What to read first

- `CLAUDE.md`: the thesis-focused guide. Start here for anything that touches the
  LaTeX source: chapter map, build command, prose conventions, and the mapping
  from each figure/table/number back to the code that produced it.
- `thesis_notes/`: current thesis narrative, conclusions, algorithms, and code
  references. These notes should not contain launch commands.
- `methods/`: current method specifications that support the thesis notes.
- `findings/`: consolidated empirical observations that are useful but not
  canonical thesis chapters by themselves.
- `operations/`: runbooks, infrastructure notes, and debugging records.
- `training/`: notes for taxonomy-aware post-training and RL/RSA recipes.
- `thesis_latex/`: the DISI LaTeX template and the thesis source itself
  (chapters, figures, tables, `biblio.bib`).
- `thesis_supporting_material/`: prior work, the style-model thesis, full-text
  papers and books, reference lists, and the literature-review ledgers.
- `thesis-TODOs/`: correction lists to apply after the author reviews a drafted
  chapter, plus the BibTeX verification log.
- `tmp/`: temporary or in-progress design notes.
- `commands.md`: personal command scratchpad and run history. It is intentionally
  not cleaned into the thesis narrative.

## Current tree

```text
docs/
  README.md
  CLAUDE.md                     # thesis-focused agent guide (LaTeX + provenance)
  commands.md
  thesis_notes/                 # numbered research findings log (NNN-slug.md)
  methods/
  findings/
  operations/
  training/
  thesis_latex/
    template-latex-lm-disi-en/
      lm_master_disi_en/        # the thesis: chapterN.tex, figures/, tables/, biblio.bib
  thesis_supporting_material/
    examples_of_thesis/         # Marco Garosi's thesis + defence slides (style model)
    full_papers_books/          # full-text PDFs of works cited and re-checked
    literature-review/          # per-paper evidence ledgers (ch2, ch3)
  thesis-TODOs/
  tmp/
```

## Maintenance rule

If a document describes current methodology, keep it under `thesis_notes/` or
`methods/` and make it agree with the code. If it is a command recipe or cluster
debugging note, keep it under `operations/` or `commands.md`. If it is stale and
its knowledge is already represented elsewhere, remove it rather than keeping a
parallel outdated explanation.

Never edit a chapter file without first checking `thesis-TODOs/pending-corrections.md`.
