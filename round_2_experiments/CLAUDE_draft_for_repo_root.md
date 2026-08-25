# CLAUDE.md (draft)

Draft for Adam's approval. If adopted, rename to `CLAUDE.md` at the repo root, where an
assistant working in this repo reads it automatically at the start of every session.

This is experimental protocol only. None of Adam's writing conventions are here, and
they are not wanted here: they govern the manuscript's prose and have nothing to say
about runs. Almost everything below is already how the experiments in this repo were
done, written down so an assistant follows it without being told each time.

## What this repo is

Experiments behind a paper on whether a confidence score can decide when a language
model should ask a clarifying question. The experiment families:

- **E-A**, the freeze test: does a deployed model update its stated forecast on evidence
  that should move it? Harness in `freeze_test/`. Not yet run on a real model.
- **E-B**, matched-confidence ask-or-answer: two question sets matched on the model's
  stated confidence, one ambiguous with annotated readings and one unambiguous but hard,
  each item run under four conditions.
- **E-C**, the between-reading variance estimator: sample answers, group them by reading,
  take the between-group variance, and test whether it predicts the gain from clarifying.

Condition names differ between the repo and the paper on purpose. The repo keeps
answer-now, oracle-clarify, self-ask, and free-choice; the paper renames them. Do not
rename anything here to match the paper.

## Provenance: every number traces to a committed file

- Every value that reaches a report or the paper derives from a committed
  machine-readable summary, with its source path recorded beside it. A number that
  exists only in terminal output does not exist.
- Every record carries model and immutable revision, prompt hashes, serving version,
  seed, node, Slurm job and step, and source-file hashes and commit.
- Report the run as performed. Claims describe the runs made and do not extrapolate to
  settings not run.

## Frozen prompts and preregistration

- Prompts are committed before the run that uses them, and verified byte-for-byte against
  that commit at run time. A recorded prompt hash does not substitute for reading the
  prompt back from the named commit.
- A non-pilot run refuses to start without its prompt commit.
- A frozen prompt that turns out to be broken gets fixed, noted, and the whole affected
  arm re-runs. Do not patch mid-run.
- Pilot items are excluded from the confirmatory run that follows, by explicit exclusion
  manifest.
- For publication-grade runs, freeze first: the confidence band and its fallback rule,
  the grading rule, the predictions, the elicitation form, exclusion rules, seeds, and
  the item population with its size. Deciding any of these after seeing match rates or
  results makes the run exploratory, whatever it is called.

## Analysis and reporting

- Intervals throughout: Wilson for proportions, percentile bootstrap with 10,000
  resamples paired within item for gains and correlations. State which.
- A Set A correlation is not interpretable without its Set B false-structure control
  beside it. Surface-form clustering manufactures apparent structure out of ordinary
  answer variation, and the control is what catches it.
- When the same samples estimate both a predictor and an outcome, the split-half
  correlation is the estimate and the same-sample figure is marked as inflated.
- Report the arm that fails beside the one that works, and keep both filtered and
  unfiltered summaries in the record.
- Do not silently drop, cap, or truncate. If coverage is bounded, say what was left out.

## Costs and item populations

Runs are cheap: a full arm is about three minutes on DGX-26 and compute cost is near
zero. The scarce resources are design time and attention. Do not size, trim, or reuse
item pools to save compute, and do not let already-paid-for samples drive design choices.
When a design question trades power or a clean item population against extra runs, take
the extra runs.

Sample reuse caused the one real design error in this project. An item pool built for
E-B's confidence matching was inherited by E-C and then by the clustering arm, so a
question about deployment got answered on the pool's hardest one to two percent. Before
running, check that the population was chosen for the question being asked.
