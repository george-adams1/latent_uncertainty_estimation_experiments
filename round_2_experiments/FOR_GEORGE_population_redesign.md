# Item population for the clustering arms: discovery at scale, prediction on a chosen sample

Part 1 is for George: what the frozen run showed, how the
design drifted, and the decisions that are yours. Part 2 is written to be handed
to Claude Code verbatim as the implementation brief; it is self-contained, so it
repeats context on purpose.

---

## Part 1: to George

**What the frozen run established.** The 2026-08-25 confirmatory clustering run
(`ec/experiment_results_1/EC_CLUSTERING_FULL_RESULTS.md`) came back clean and
answered its question on the items it ran: with annotation-blind clustering, the
E-C correlations fall to the answer-string baseline level, every Q and QS interval
includes zero, and the agreement audit locates the failure in the lister. Given a
usable interpretation list, assignment agreed with oracle labels on 84-94% of the
uniquely matched responses, but the pipeline recovered both annotated readings on
only one item per subject model. Reading discovery, not assignment and not the
estimator, is the bottleneck. That result stands as the record on its item slice
and nothing below overwrites it.

**How the design drifted.** This is the part I got wrong, and it is worth spelling
out so the fix makes sense.

The item pool was built for E-B. Testing whether confidence can route
clarification requires ambiguous and unambiguous items at matched confidence,
otherwise the confidence baseline loses for an unfair reason. So E-B screened the
2,956 two-reading AmbigQA candidates through the 50-60% confidence band, which
kept roughly 1 to 1.5% of them, per subject model. That was the right design for
that hypothesis, and E-B keeps it.

E-C then reused E-B's items because the 32-sample batches were already on disk,
and the clustering arm reused E-C's items for the same reason. Each reuse looked
free and saved a run. The effect was that a question about deployment, whether a
system can discover readings on incoming questions, was answered on 25 to 39
items per model that had been selected, by construction, for ambiguity subtle
enough to leave the subject model near coin-flip confidence. The pool is close to
adversarial for a lister, the per-model screening means the two subject models
were scored on different items, and the sample is too small to bound anything:
at n of about 40, a true correlation of 0.3 carries a 95% interval from below
zero to 0.56 (Fisher z, verified). So the frozen run shows discovery fails on the
hardest slice, and cannot say what happens on the population the deployment claim
is actually about.

The root cause was a wrong premise about cost. We were treating runs as
expensive and sample reuse as the prudent default, when your arms take about
three minutes on DGX-26 and the compute cost is near zero. Adam has recorded the
corrected premise in the project state: the scarce resources are design time and
attention, and item pools are not to be sized or reused to save compute. This
note is the first application.

**The correction.** The clustering work splits into two experiments with
separately chosen populations, because the frozen run conflated two questions
with very different data needs:

- **Experiment D (discovery).** Can the fixed lister find the readings at all?
  This needs no subject-model samples and no gain measurement, so it runs on the
  full 2,956-item two-reading pool plus a large unambiguous control set. One
  lister call per item, plus an evaluation-only judge stage. This is the question
  the negative result is really about, and at full-pool scale the intervals
  become tight (width about 0.07 on a correlation, and comparably tight Wilson
  intervals on detection rates).
- **Experiment P (prediction).** Does the estimator's correlation with realized
  gain hold under inferred labels? This needs the expensive part, three
  32-sample batches per item, so its population is a deliberate choice rather
  than an inheritance: unscreened by confidence, sized by power. At n = 300, a
  true r of 0.3 gets interval [0.19, 0.40] and a true r of 0.6 gets
  [0.52, 0.67] (verified), so oracle-level and collapsed-level correlations
  separate cleanly. The full pool is about 284k generations per subject model
  for the sampling stage, which is hours on your node, so full pool is on the
  table for the DGX models if the wall-clock suits you.

D also buys P's design for free: once D has per-item discovery outcomes, P's
sample can be stratified by discovery difficulty, and the paper can report where
on that spectrum the estimator's correlation holds under inferred labels.

**Decisions that are yours before running:**

1. **Control set for D.** Source and size for the unambiguous items. Building it
   the way Set B was built keeps the comparison familiar; a larger set is better
   and costs nothing. Aim for at least the size of the ambiguous pool.
2. **Judge model for the recall audit.** The judge matches inferred
   interpretation descriptions to annotated readings, after all inference, so it
   may see annotations. Same Llama-70B or a stronger model, your call; the
   prompt freezes before the run either way, and a human spot-check of a random
   sample of judged pairs is part of the protocol.
3. **QS.** The frozen run showed QS adds Set B false positives without adding
   Set A detection. My recommendation is Q only for D (QS needs samples, which D
   does not otherwise use) and both variants in P where the samples exist
   anyway. If you keep QS anywhere, it stays a recorded secondary variant.
4. **P's population per model.** Full pool versus stratified n on the DGX
   models is a throughput call; for the frontier API model the calls are the
   budget, so a stratified subset of a few hundred Set A items is the realistic
   form. Decide n before sampling starts and record it in the prereg.
5. **File locations**, mirroring `scan_results/` and `experiment_results_1/`
   conventions as you prefer.

**Where this sits in the plan.** The model-slate priorities are unchanged. D is
nearly free and can run ahead of everything on the current clusterer. P is the
publication-grade E-C's clustering arm and runs on the new slate
(Qwen3-32B, Llama-70B, frontier), with its population decision folded into the
prereg freeze alongside the confidence band, grading rule, elicitation form,
exclusions, and seeds. The paper rescopes the frozen 2026-08-25 result to its
slice: inferred labels fail on the matched-confidence subset, stated with the
selection named.

**One addition since this note was written.** A third labeling arm joins the
submission path as item 4 of the queue in `paper2_plan.md`, and it is not D and
not P. Your lister discovers readings from the question text; the alternative
discovers them from the answers, by clustering the 32 saved samples under
bidirectional entailment. That is how semantic entropy is computed, and the
semantic-entropy baseline was already owed to the reception pass, so one
clustering pass yields both numbers: the entropy over cluster masses for the
baseline, and the between-cluster variance of the answer indicator for the
estimator under inferred labels. It reuses the saved batches the way your Q and
QS arms did, runs through `estimate_from_labels` unchanged, and reuses the
split-half bootstrap and the maximum-overlap agreement audit as they stand.

What is yours to decide: the clusterer, an NLI model or an LLM prompted for
entailment, committed before the run as the lister prompts were; and the pairwise
budget over 32 samples per prompt, where the greedy agglomerative shortcut from
the semantic-entropy papers is the cheaper form if the full pairwise matrix is
too much on the node. Both go in the prereg freeze.

Report it beside the oracle, Q, and answer-string columns with the Set B
diagnostics alongside, since a Set A correlation says nothing without the
false-structure control. The bar is a Set A interval clearing zero with Set B
observed categorical variance near the oracle arm's 0.0761 and 0.0406 rather than
the answer-string arm's 0.5680 and 0.2396. Clearing it puts the inference-time
claim back in the paper. Failing it is also worth reporting, since it shows the
discovery failure is not an artifact of asking a model to narrate
interpretations. The arm is off the gate list either way and cannot delay the
submission.

---

## Part 2: brief for Claude Code

You are working in the repo
`github.com/george-adams1/latent_uncertainty_estimation_experiments`. The task
adds two experiments around the existing E-C inference-time clustering arm. Read
this brief fully, then explore `ec/clustering.py`, `ec/run_clustering.py`,
`ec/prompts/`, and `ec/experiment_results_1/` before writing code.

### Background you need

The E-C experiment computes, per question, a between-reading variance $\hat V_B$
from 32 sampled answers clustered by reading. The clustering arm of record
(frozen run, 2026-08-25, `ec/experiment_results_1/`) inferred clusters with a
fixed Llama-3-70B-Instruct pipeline: a lister proposes 1 to 4 interpretations of
the question, an assigner maps each sampled answer to an interpretation or
`none`. Prompts were frozen at commit `79554bc`. The run showed that inferred
labels lose the oracle correlation, and its agreement audit located the failure
in the lister: assignment is accurate given a usable list, but both annotated
readings were recovered on only one item per subject model.

That run's items were inherited from E-B's confidence-band screening, roughly 1
to 1.5% of the available pool, selected for subtle ambiguity. The two
experiments below re-ask the question on deliberately chosen populations. The
existing frozen results are not overwritten, reprocessed, or reinterpreted; the
new experiments sit beside them.

### Experiment D: reading discovery at scale

**Population.** All 2,956 Set A two-reading AmbigQA candidates (the full
screening pool, before any confidence filtering), plus an unambiguous control
set whose source and size George supplies. Items that appeared in the E-B/E-C
runs or the clustering pilot are included but flagged in the records.

**Inference stage.** For each item, one call to the frozen Q lister: question
text only, prompts byte-identical to commit `79554bc`, verified the same way
`run_clustering.py` verifies prompt-commit equality. Same clusterer model and
immutable revision as the frozen run, temperature 0, seed 0, full provenance
per record (model, revision, prompt hashes, server version, node, Slurm
job/step, source commit). No assigner calls. No subject-model samples. The
lister never sees annotations, gold answers, rewrites, or the set label. QS is
omitted unless George directs otherwise; if included, it is a separately
recorded secondary variant and requires sampling.

**Audit stage, after all inference.** A judge (model chosen by George, prompt
frozen before the run) receives one inferred interpretation description and one
annotated disambiguated question and returns match or no-match. The judge is
evaluation-only, so annotation visibility is allowed here and only here. Score
each ambiguous item by maximum bipartite matching between inferred descriptions
and the two annotated readings, so one vague description cannot claim both
readings. Per item: readings recovered (0, 1, or 2) and count of inferred
descriptions matching no annotated reading. Write out a random sample of judged
pairs (size per George, at least 50) to a file for human spot-check.

**Metrics, with Wilson or bootstrap 95% intervals throughout:**

- On ambiguous items: distribution of readings recovered; rate of full recovery;
  rate of multi-interpretation lists.
- On control items: multi-interpretation rate (the false-positive control).
- Discrimination: treating "returned more than one interpretation" as a binary
  ambiguity classifier, sensitivity and specificity; and AUC using the
  interpretation count.
- Everything above stratified by any cheap item covariates available (question
  length, wh-word), as a secondary table.

### Experiment P: correlation preservation on a chosen sample

Runs per subject model of the publication-grade slate, after the prereg freeze.
This is the clustering arm of the publication-grade E-C, redesigned in one way:
the item population is sampled from the two-reading pool without confidence
screening. George fixes n and the sampling rule (simple random, or stratified by
Experiment D's discovery outcome) in the prereg before sampling starts. Sizing
guidance, verified with a Fisher z computation: n = 300 gives a 95% interval of
[0.19, 0.40] at r = 0.3 and [0.52, 0.67] at r = 0.6, so the collapsed and
oracle regimes separate; the full pool costs about 284k generations per subject
model for sampling and tightens the interval width to about 0.07.

Per item, the pipeline is the existing one, unchanged: three 32-sample batches
(ambiguous prompt, each fixed-reading rewrite) from the subject model at
temperature 1; lister and assigner with frozen prompts; $\hat V_B$ under oracle,
Q, QS, and answer-string labels through the identical estimator; realized gain
as fixed-reading gain averaged over both readings; split-half correlations with
the existing bootstrap settings. The annotation firewall is unchanged: the
clusterer never sees annotations, gold answers, rewrite text, or condition
names, and annotation-derived labels are read only after all clustering calls.

### Rules (protocol, not suggestions)

- Prompts frozen before each run: the lister and assigner prompts are the
  `79554bc` set reused byte-identically; the judge prompt and any new prompt are
  committed before their run and unchanged after results are seen. A broken
  frozen prompt gets fixed, noted, and the whole affected arm re-runs.
- Population and n for Experiment P are fixed in the prereg before sampling.
- Every reported value derives from a committed machine-readable summary with
  its source path recorded, matching the repo's convention.
- Record-level provenance as in the frozen run: model, revision, prompt hashes,
  seeds, node, job/step, source file hashes and commits.
- Do not modify anything in `ec/experiment_results_1/`.

### Outputs

1. Experiment D: per-item JSONL (item id, set label, flags for prior use in
   E-B/E-C/pilot, lister output, judge matchings, recovery score), a summary
   JSON with every metric above, the judged-pair spot-check file, and a short
   results markdown sourcing every number from the summary by path.
2. Experiment P: the same output set the frozen clustering run produced
   (per-item JSONL, summary JSON, results markdown), one per subject model,
   plus the prereg file recording population, n, sampling rule, and seeds.

### What the results mean (so the summaries say the right thing)

- Experiment D's headline is the recovery distribution on the full pool against
  the false-positive rate on controls. If full recovery is rare everywhere, the
  frozen run's negative generalizes and the paper says discovery fails broadly.
  If recovery is common outside the matched-confidence slice, the frozen
  negative is a statement about the hardest items, and the paper says the
  bottleneck is concentrated where ambiguity is subtlest. Either way the claim
  gets its population stated.
- Experiment P's headline is oracle versus inferred correlation at a sample size
  where the difference is estimable. Report the loss plainly; do not smooth it.
- The control-set false-positive rate bounds the cost of deploying the lister as
  an ambiguity screen. Report it beside the sensitivity, not in isolation.

---

*The plan entries pointing here are in `document rules and plans/paper2_plan.md`
and the model slate note `FOR_GEORGE_model_slate.md`. The corrected compute
premise is recorded in `document rules and plans/STATE.md` (2026-08-25).*
