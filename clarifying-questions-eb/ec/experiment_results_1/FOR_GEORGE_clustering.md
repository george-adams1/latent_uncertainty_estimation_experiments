# Inference-time clustering arm for E-C

**Implementation status (2026-08-25): complete.** The frozen confirmatory run
used Llama-3-70B as the fixed clusterer for Qwen3-8B and Llama-3-70B saved
responses. See `clarifying-questions-eb/EC_CLUSTERING_FULL_RESULTS.md` and the
machine-readable
`clarifying-questions-eb/ec/experiment_results_1/ec_clustering_full_*`
artifacts. The remainder of this file is the original pre-implementation brief.

Two parts. Part 1 is for George: what changed and why, and the decisions that are
yours. Part 2 is written to be handed to Claude Code verbatim as the implementation
brief; it is self-contained, so it repeats context on purpose.

---

## Part 1: to George

**Why this exists.** The estimator's number is inherited from the clustering. In the
E-C runs of record, clustering was alias matching against AmbigQA's *annotated*
readings, which is benchmark information a deployed system does not have. So the
0.58-0.83 correlations validate the estimator *given correct reading labels*, and no
run yet shows the pipeline working when the system must identify the readings itself,
which is the only form usable at inference time. The failure mode is concrete:
clustering by distinct answer strings gives a large value on unambiguous items (a
worked example: eight scattered guesses give 0.171 against a two-camp 0.246), which
is essentially the undifferentiated variance that predicted nothing in Table 2. A
real clusterer must use the discriminating fact that "Atlanta" and "Tbilisi" answer
*different questions*, while "1901" and "1899" are competing answers to the *same*
question.

**What we are adding.** One clustering arm and one audit on top of the confirmatory
E-C already planned (priority 2 in `FOR_GEORGE_model_slate.md`); no new sampling is
required if the existing 32-sample records are reused. The full spec is Part 2.

**Decisions that are yours before running:**

1. **The clusterer model.** My recommendation: one fixed model serves as the
   clusterer for every subject model, so the arm is comparable across the slate; a
   mid-size open-weight you can run locally is fine. Using each subject model as its
   own clusterer is a separate, interesting question, but it confounds the
   comparison, so if you want it, make it a second recorded variant rather than the
   primary.
2. **Pilot, then freeze.** Iterate the lister prompt on a small subset first,
   checking the lister's recall against the annotated readings, because the evidence
   we have predicts under-detection: AmbigQA ambiguity is subtle, the blind
   confidence elicitation collapsed to 85-95% on these very questions, and even
   Llama's free choice asked on only 41% of ambiguous items. A weak first prompt
   would otherwise either doom the arm or tempt post-hoc edits. After the pilot,
   the prompts are frozen: committed before the full run and not edited after
   results are seen. If a frozen prompt turns out broken, fix it, note the change,
   and re-run the whole arm. Keep the pilot subset out of the reported analysis or
   report it as pilot, either way marked.
3. **Where it lands in the repo.** Part 2 proposes file locations that mirror your
   existing `scan_results/` layout; adjust to taste, but keep every reported value
   derivable from a committed machine-readable summary, as with everything else.

**Stakes.** The paper already states the kill condition this arm tests: if blinded or
inference-time clustering drags the correlations toward the raw-variance baseline,
the estimator loses its margin over existing diagnostics. Either outcome is
reportable; not running it leaves deployability unmeasured, and the mock-review
panel flagged exactly this gap.

---

## Part 2: brief for Claude Code

You are working in the repo
`github.com/george-adams1/latent_uncertainty_estimation_experiments`. The task is to
add an **inference-time clustering arm** to the E-C experiment. Read this brief
fully, then explore the repo's existing E-C pipeline before writing code.

### Background you need

The E-C experiment measures, per ambiguous question, a "between-reading variance"
$\hat V_B$ computed from 32 sampled answers (temperature 1) that were clustered by
which reading of the question each answer matches. In the existing runs the
clustering is **oracle**: answers are alias-matched against the two annotated
readings that AmbigQA provides per item. The records are
`scan_results/ec_*_results.jsonl` with summaries in
`ec_*_results.jsonl.summary.json`, and `ec/split_half.py` computes the reported
split-half correlations between $\hat V_B$ and the realized clarification gain. Set A
items are ambiguous (two annotated readings); Set B items are unambiguous controls.

The new arm answers: does the estimator still predict the realized gain when the
readings are identified by an LLM pipeline **with no access to the annotations**?

### What to build

Three stages, run per item over the *existing* sampled answers (reuse the committed
32-sample records; do not resample):

1. **Interpretation lister, two recorded variants.** A fixed LLM (model name
   supplied by George; record it and its version in every output) returns a list of
   plausible interpretations of the question, from 1 up to a cap of 4. Returning
   exactly one interpretation is a legal and expected output for unambiguous
   questions. The prompt must not hint at an expected number of interpretations and
   must not include anything from the dataset annotations. Run both variants on
   every item and keep their outputs separate end to end:
   - **Variant Q (question only):** the lister sees the question text and nothing
     else. This is the pure form of the assumption; its expected failure mode is
     missing subtle ambiguity (false negatives on Set A).
   - **Variant QS (question plus samples):** the lister additionally sees the 32
     sampled answers and is asked whether they reflect different readings of the
     question. This attacks the recall problem (two answer camps are a loud hint
     that two readings exist); its expected failure mode is inventing
     interpretations to explain ordinary scatter (false positives on Set B). The
     Set B cluster-count control is the adjudicator between the variants, and that
     comparison is itself a reported result.
2. **Assigner.** For each sampled answer, the same fixed LLM assigns the answer to
   one of the listed interpretations, or to "none of them". Its context is the
   question, the interpretation list, and the single answer text. It never sees the
   annotated readings, the gold answers, the rewrites, or the other samples.
3. **Estimator.** Compute $\hat V_B$ exactly as the existing E-C code computes it
   (same formula, same statistic $h$, same conventions for empty clusters and
   unassigned samples), changing only the cluster labels: the assigner's labels
   replace the oracle alias labels. Do not introduce a new estimator formula; if the
   existing computation is entangled with alias matching, refactor so the same
   function accepts either labeling.

Also compute one cheap baseline arm for the same items: **answer-string clustering**,
where each distinct normalized answer string is its own cluster. This is expected to
fail on Set B (it manufactures disagreement where there is one reading) and exists to
demonstrate that failure with the same code paths.

### Rules (these are protocol, not suggestions)

- The clusterer prompts are frozen: committed before the run, unchanged after
  results are seen. If a prompt must change, the change is committed with a note and
  the whole arm re-runs.
- The clusterer never sees gold answers, aliases, rewrites, or annotations, in any
  stage.
- Set B is processed identically to Set A, through the same lister and assigner. Do
  not special-case it: the number of interpretations the lister returns on Set B is
  itself a measured outcome.
- Every reported value must be derivable from a committed machine-readable summary
  file, with the source path recorded, matching the repo's existing convention.
- Record: clusterer model and version, prompt file hashes, seeds if any, and the
  commit of the sample records consumed.

### Outputs

1. Per-item records (JSONL, one file per subject model, mirroring the existing
   `scan_results/` naming): question id, set (A/B), the interpretation lists from
   both lister variants, per-sample assignments under each (including "none"),
   cluster sizes, $\hat V_B$ under (a) oracle alias labels, (b) inference-time
   labels from variant Q, (c) inference-time labels from variant QS, (d)
   answer-string labels.
2. Summary JSON per subject model with: split-half correlations of realized gain
   against each of the four $\hat V_B$ variants (reuse `ec/split_half.py`,
   both half-assignments, same bootstrap settings as the record); the cluster-count
   distribution of each lister variant on Set A and on Set B separately; and the
   Set A agreement audit per variant: rate at which inference-time assignments
   match the oracle labels, plus counts of samples assigned "none" and of samples
   whose answer matched neither or both alias sets under the oracle rule.
3. A short results markdown in the repo summarizing the three-way comparison per
   model, with every number sourced from the summary JSONs by path.

### What the results mean (so the summary says the right thing)

- The headline comparison is inference-time versus oracle: how much correlation is
  lost when the annotations are dropped. Report the loss plainly; do not smooth it.
- The Q-versus-QS comparison is the recall-precision trade, measured: Q is expected
  to miss subtle ambiguity (Set A recall against the annotations is its weak spot),
  QS to over-detect (Set B cluster counts are its weak spot). Report both sides;
  which variant wins is an empirical result, not a prior.
- The Set B cluster-count distribution is the honesty control: a lister that returns
  more than one interpretation on many Set B items is manufacturing ambiguity, and
  the resulting nonzero $\hat V_B$ values on Set B quantify the false-positive cost.
- The answer-string arm is expected to produce large $\hat V_B$ on Set B; its
  purpose is to show the failure mode the reading structure prevents.
- Do not overwrite or reinterpret the oracle-arm results of record; the new arm sits
  beside them.

---

*The plan entry pointing here is in `document rules and plans/paper2_plan.md`,
Step 2, E-C.*
