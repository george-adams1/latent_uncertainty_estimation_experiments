# Start here: what we are doing and where it stands

Written 2026-08-25, plain language, results rather than proofs. The math is in the paper
and nothing below depends on reading it.

## The short version

The clustering run came back negative and it changed what the paper claims, so the plan
moved. What is wanted from George: run Experiment D first, since it is nearly free and
decides how the paper states that negative; then the publication-grade E-B with E-C
beside it; then the answer-side clustering arm. The preregistration has to freeze this
week for the numbers to be settled against a September 25 deadline. Decisions that are
his are listed at the end. Everything in between is the reasoning behind that order.

## The problem

A chatbot gets a question it could read more than one way. It can answer now, or it can
ask the user what they meant and answer after. Deployed systems make that choice by
thresholding a confidence number: answer when confident, ask when not.

Our claim is that this is the wrong instrument, and not because the number is
miscalibrated. Consider two situations that both produce "55 percent confident":

- The question has two readings and the model knows the answer to each perfectly. It
  just does not know which one you meant.
- The question has one clear reading and the model half-knows the answer.

Asking a clarifying question is worth a lot in the first case and worth nothing in the
second. The confidence score is the same number in both. It measures how much
uncertainty there is; the decision needs to know where the uncertainty sits. Squeezing
the belief into one number is the step that destroys the distinction, so no amount of
better calibration recovers it.

## Terms used here

- **Set A**, the ambiguous questions, drawn from AmbigQA, each with exactly two annotated
  readings and an answer for each. **Set B**, the control: single-reading trivia
  questions that are simply hard, drawn from TriviaQA.
- **The pool** is the 2,956 AmbigQA candidates with exactly two readings, before any
  filtering. **The band** is the 50 to 60 percent stated-confidence window that items had
  to land in to enter the matched sets, which kept about 1 to 1.5 percent of the pool.
- **The four conditions** each item runs under: *forced answer* (must answer immediately),
  *forced ask* (must write a clarifying question, gets a reply, then answers), *free
  choice* (decides for itself), and *disambiguated* (handed the annotated rewrite, the
  ceiling on what resolving the reading is worth). Run records keep the older names
  answer-now, self-ask, free-choice, and oracle-clarify.
- **Between-reading variance** is the estimator this paper proposes: sample answers,
  group them by reading, and measure how much the groups disagree. Called the estimator
  throughout.
- **Oracle labels** group answers using the dataset's annotations. **Inferred labels**
  make the system work them out for itself, which is what deployment would require.
- **E-A, E-B, E-C** are the three experiment families, described below, and **D** and
  **P** are the two new ones.

## What we found

### The score does not track the value of asking (E-B)

We built two sets of questions matched on the model's own stated confidence, so the
score cannot tell them apart by construction. Then we ran each item under the four
conditions above.

Resolving the ambiguity was worth about 13 and 16 accuracy points on Set A across the two
models, and zero on Set B by construction, since there is no reading to resolve. Llama
captured 9 of its 16 available points by writing its own clarifying question, against 2
points on Set B, and it chose to ask on 41 percent of Set A items versus 9 percent of Set
B. Confidence did not predict which items benefited.

Qwen is the interesting case. The value was there, 13 points available, and it captured
none of it. We cannot read much into that beyond the obvious, because it scored zero on
Set A whether it asked or not, and a set pinned at zero cannot give a fair within-model
test. What it does show is that capturing available value takes competence, and that this
is a separate matter from whether confidence can see the value.

### Something else does track it (E-C)

Sample the model's own answers to the ambiguous question, group the answers by which
reading each one commits to, and measure how much the groups disagree with each other.
That quantity is the between-reading variance. When the samples split into two camps
that answer differently, a clarifying reply settles which camp is right and asking pays.
When they scatter with no camp structure, no clarifying question helps.

The disagreement is not a heuristic correlate of the value of asking. Under a standard
scoring rule it equals the improvement a perfect clarification would give, exactly. So it
comes out in the same units as the cost of asking, and the two can be compared directly.

It works empirically too: it predicts which items benefit at correlations of 0.58 to
0.83 across the two models, while the same samples stripped of the reading structure
predict almost nothing, and confidence predicts nothing.

### The catch we just found

All of that used oracle labels. When George made the system infer the readings itself,
the correlations dropped to the level of the naive baseline and every interval included
zero.

The agreement audit tells us where it broke, which is the useful part. Noticing that a
question is ambiguous works reasonably. Sorting answers into a list of readings, once
you have a good list, works well, at 86 to 90 percent agreement. What fails is coming up
with the right pair of readings in the first place: the pipeline recovered both annotated
readings on 1 of 25 items and 1 of 39. So the bottleneck is reading discovery, not the
sorting step and not the estimator.

One qualifier matters. That test ran on the items screened into the band, the hardest 1
to 2 percent of the pool, selected for ambiguity subtle enough to leave the model near a
coin flip. That is close to a worst case for finding readings, and we do not yet know
what happens on ordinary ambiguous questions.

## Where the paper stands

It claims the estimator is the right quantity and is validated under oracle labels, and
it reports inference-time reading discovery as an open problem with the item population
stated. That matches the evidence, and it is already written that way.

If reading discovery can be made to work before the deadline, the claim gets stronger and
we say so. That possibility is what several of the experiments below are for.

## Why each experiment runs

Each one answers a question. If an experiment does not answer a question we care about,
it should not run.

- **Experiment D, reading discovery at scale.** Is discovery broadly broken, or only
  broken on the subtle slice we happened to test? One cheap call per item over the whole
  pool plus a control set answers it, with no model sampling at all, and it decides how
  the paper states the negative. Cheapest, and first.
- **Publication-grade E-B.** Does the confidence result hold on better models, with the
  protocol frozen in advance and more items? This is the central claim and its current
  evidence is two mid-size models with about 30 and 44 items. We commit in advance that
  if a frontier model's confidence does separate the sets, the claim narrows to mid-size
  open-weight models and we say so.
- **E-C across the same models.** Does the estimator result replicate? Cheap, and the
  table repeated across models is its strongest form.
- **Answer-side clustering.** The failed pipeline tried to find readings by reasoning
  about the question. The alternative finds them from the answers, by grouping samples
  that mean the same thing. This is also how semantic entropy is computed, and semantic
  entropy is the comparison we owed reviewers anyway, so one run produces both numbers.
- **Reversed-order control.** Closes a small caveat about a repeat-batch asymmetry on
  Qwen's Set B.
- **Experiment P, prediction on a chosen sample.** Does the estimator keep its
  correlation under inferred labels, measured on a population picked for that question
  rather than inherited from E-B's confidence screening? It needs the expensive part,
  three 32-sample batches per item, so it waits on what D reports.
- **E-A, the freeze test.** Does a real model actually show the behavior the theory
  describes, ignoring evidence that should move its forecast? Informative either way, and
  explicitly not required for the submission.

## One gap worth flagging

There is a mismatch between what the estimator has been validated against and what we
ultimately claim, and it does not appear to have been noticed.

E-C measures its outcome the oracle way: replace the ambiguous question with the
dataset's disambiguated rewrite, average over both readings, and see how much accuracy
improves. So when we say the estimator predicts realized gain at 0.58 to 0.83, the gain
being predicted is a benchmark quantity that uses annotations.

What a deployed system actually gets is different: the model writes its own clarifying
question, a user replies, and it answers. That is E-B's forced-ask gain, and the paper is
explicit that the two are different estimands and should not be differenced.

Nobody has checked whether the estimator predicts the deployed gain. That is the closest
thing to the practical claim, and it may be free: E-B and E-C ran on the same matched
items, so the between-reading variance from E-C's 32 samples can be correlated against
E-B's per-item forced-ask outcome with no new runs. The earlier 8-sample version of the
diagnostic, collected during E-B, pointed the right way but was too small to conclude
from, and 32 samples is the real test.

Power will be poor, since per-item forced-ask outcomes are close to binary at these
sample sizes, so a null would not mean much while a positive would mean a lot. And it
only works for models that capture value from their own questions at all, which rules
out Qwen3-8B, whose Set A accuracy was pinned at zero.

Worth an hour before the publication-grade runs are designed, because if it looks
promising the deployed-gain correlation should be a preregistered outcome of the big run
rather than an afterthought. Worth settling between the three of us at the meeting; it is
a suggestion, not a settled item.

## The rest of the package

- **`paper2_plan.md`**, the working plan. Its claims-to-evidence table (C1 to C5) maps
  each claim the paper makes to the experiments supporting it and their status, and the
  queue below it is the run order. The rule attached to the table: no experiment enters
  the queue without naming the claim it serves.
- **`FOR_GEORGE_population_redesign.md`**, the brief for Experiments D and P, including
  how the item population drifted and why it splits in two. Its closing section covers
  the answer-side arm.
- **`EC_HYPOTHESIS_SUMMARY.md`**, the E-C record and confirmatory checklist, updated so
  its clustering-audit item reflects the frozen run.
- **`CLAUDE_draft_for_repo_root.md`**, draft experimental conventions for the repo root:
  provenance, frozen prompts, preregistration, and what gets reported, collected from
  briefs where they are currently scattered. Mostly a written-down version of what these
  experiments already do, so that an assistant working in the repo follows it without
  being told each session. If Adam approves, rename it to `CLAUDE.md` at the repo root.
  Do not commit it before he confirms.

## Over to George

Run order is in the list above and in the plan's queue. Decisions that are his before
anything runs: the control set for Experiment D, the judge model for its recall audit,
the entailment clusterer and its pairwise budget for the answer-side arm, P's population
and size if P runs, and file locations. The preregistration freezes before the
publication-grade runs, and that needs to happen this week for numbers to be settled by
September 12 to 15 against the September 25 deadline.

## Before posting these

The experiments repo is public. These notes name both authors, the paper title, the
venue, and the submission dates, and ICLR 2027 is double blind with desk rejection on
identity leaks. A private repo is the safer home and works identically for an assistant
reading the files. Worth settling before these go into a public one.
