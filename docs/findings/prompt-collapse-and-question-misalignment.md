# Prompt Collapse and Question Misalignment

This note consolidates two project findings: the "concise" prompt collapse and
the OVEN question-template mismatch with specific ground-truth entities.

## Concise prompt collapse

The original concise prompt strongly encouraged the model to answer "I don't
know" when uncertain. In OVEN open-world entity recognition this instruction
dominated generation: many rollouts collapsed to abstention-like answers instead
of exploring plausible entity names.

The practical result was that prompt wording controlled the measured behavior as
much as model capacity. Removing or weakening the "I don't know" instruction
produced a more useful sampling distribution for pass@k and taxonomy analysis.

Current thesis relevance:

- prompt instructions can alter coverage estimates;
- pass@k should be interpreted as a property of the model plus decoding prompt;
- the `concise_no_idk` condition is the cleaner baseline for coverage analysis.

## Question-template misalignment

OVEN questions can be too generic relative to the ground-truth entity. For
example, a question may ask for a broad visual category while the ground-truth
answer is a specific Wikipedia/Wikidata entity.

This creates two evaluation problems:

- a model can give a reasonable category-level answer that is not the target
  entity;
- a judge can over-credit under-specific answers unless the prompt makes the
  granularity boundary explicit.

The aligned-data pipeline mitigates this by using taxonomy information to create
questions that are closer to the expected answer granularity, such as asking for
the relevant parent class rather than using an overly generic template.

## Current implications

- Use aligned OVEN rows for the main thesis evaluations.
- Report prompt variant and question-construction policy with every run.
- Treat under-specific answers as a central failure mode, not as a minor parsing
  artifact.
- Use taxonomy evidence in judge prompts to expose the leaf/parent/grandparent
  distinction without making parent-only answers correct.

