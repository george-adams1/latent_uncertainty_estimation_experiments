# Paper 2 plan: Confidence Cannot Route Clarification

Paper 2's main document is `LLM UQ latex/confidence_cannot_route_clarification.tex`,
"Confidence Cannot Route Clarification, and What Can". Authors Adam M. Oberman and
George Adamopoulos (LawZero), with George running the experiments. Paper 1,
`TU arxiv/collapsing_uncertainty.tex`, is off the publication path; the operator
results it holds are restated and proved in this document's Appendix A, so the
submission is self-contained. The retired draft `clarifying_questions_design.tex` is
in `LLM UQ latex/_archive/`.

Venue: ICLR 2027. Abstracts September 18, papers September 25, 2026, both AOE. Nine
pages of main text at submission, strictly enforced, appendices unlimited, double
blind. AISTATS 2027 is the fallback if the publication-grade E-B slips.

## What is live

Settled by Adam 2026-08-25: the manuscript is paused and the experiments are the only
live work. Held until the pause lifts: the referee pass, compression, the theory
additions, and venue formatting. Live: the experiment queue below, and the design and
preregistration work that precedes it.

The draft is left coherent rather than mid-edit. The rescope is applied, it compiles
clean at 24 pages, and no site carries a claim the record contradicts, so the pause
costs nothing that has to be redone. It also costs nothing on the gate, since the
standing decision already had one referee pass covering the compressed text and the
results together.

Two dates do not move with the pause. An ICLR abstract must be registered by September
18 (AOE) for a paper to follow on September 25, and the abstract text already exists, so
that is a submission action on a fixed date rather than writing work. Working back from
September 25, the referee pass, compression, and the fresh pass after compression need
about ten days, which puts the settling of experimental values around September 12 to
15. Past that, AISTATS 2027 is the operative target.

The claims table below stays as the check on the experiments: it is what says which
claim each run serves, and it is where a result lands when it arrives.

## Claims and the evidence for them

The paper rests on the claims below. Every experiment in the queue serves one of them,
and no experiment enters the queue without naming the claim it serves. The table is
also the check that a result which lands has somewhere to land: the inference-time
clustering negative of 2026-08-25 fired a kill condition the draft had written down in
advance, and it went a day without reaching the draft because no document connected the
two.

| Claim | What it says | Evidence | Status |
|---|---|---|---|
| C1 | Stated confidence does not separate the matched sets, and does not predict realized clarification gain item by item | E-B | Exploratory, two mid-size open-weight models, $n = 30$ and $44$. Publication-grade run open |
| C2 | No score can: the ask-or-answer decision does not factor through the collapse map | thm:factor, on thm:freeze, prop:suff, prop:miscal | Proved. Statements live pending the referee pass. E-A would test the mechanism in a deployed model and is not a gate |
| C3 | Between-reading variance equals the Brier value of a reading-revealing reply, and the plug-in estimator is consistent for it under Assumption 1 | prop:reveal, prop:consistent | Proved. Statements live pending the referee pass. Debiased form open, a TODO in the tex |
| C4 | The estimator predicts realized clarification gain item by item | E-C with oracle alias clustering | Exploratory, all four directional predictions held in both models. Confirmatory run open |
| C5 | The estimator is computable at inference time, so the procedure is runnable | E-C inference-time clustering arm | Not supported on the items tested (frozen run, 2026-08-25). Rescoped and applied to the tex 2026-08-25; reading discovery is now a reported finding, Section 4.4. The answer-side repair arm is queue item 4 and could reclaim the claim |

C1 and C2 are the paper's negative half and C3 and C4 its positive half. C5 was the
practical payoff, and it is the claim that moved.

## The estimator's claim, rescoped 2026-08-25

The frozen inference-time clustering run is the record on this question: repo
`github.com/george-adams1/latent_uncertainty_estimation_experiments` at commit
`51ff250`, results in `ec/experiment_results_1/EC_CLUSTERING_FULL_RESULTS.md`, values
derived from `ec_clustering_full_qwen3_8b.jsonl.summary.json` and
`ec_clustering_full_llama3_70b.jsonl.summary.json`. Prompts frozen at commit `79554bc`
before the run, pilot IDs excluded, leaving 25 Set A items for Qwen3-8B and 39 for
Llama-3-70B-Instruct.

With annotation-blind labels the same-sample Set A correlation falls from $0.804$ to
$0.236$ (Q) and $0.209$ (QS) on Qwen responses, and from $0.569$ to $0.298$ and $0.130$
on Llama responses. Every Q and QS interval includes zero, in the same-sample and both
split-half analyses. The oracle arm stays positive, so the run does not refute the
estimator given correct reading clusters; it locates the failure before the estimator.

Adopted for the submission: the paper claims the estimator identified and validated
under reading labels, and reports inference-time discovery as a measured open problem
rather than as a limitation in passing. Contribution 4 changes from "the procedure, now
runnable" to the conditional form, and the discovery result is reported as a finding
with its item population named. The reason for taking this route over repairing
discovery first is the calendar, not a judgment that repair is out of reach: repair is
open research and the paper deadline is thirty-one days out.

The repair route stays live. If a repair arm restores a correlation distinguishable
from the answer-string baseline before compression, the inference-time claim can be
reclaimed and contribution 4 restated in its stronger form. Nothing in the rescope
forecloses it, and the design note below records which repair to try first.

The rescope was applied to the tex on 2026-08-25 and the draft compiles clean at 24
pages. The referee pass already due covers it. The sites, all now edited:

- Contribution 4, the "now runnable" item in the contributions list.
- Section 4.2, the sentence sending the clustering audit to the confirmatory checklist,
  which the frozen run has now executed.
- Section 5, "every quantity is observable" in the end-to-end paragraph.
- Section 6, the kill condition stated in advance for the clustering audit, which now
  reports a result instead of a contingency.
- Section 4.3, which gains the clustering arm's numbers.
- The introduction's paragraph on the estimator, where the guarantee's conditionality is
  described as untested.
- The Section 4.3 TODO comment's path for `EC_HYPOTHESIS_SUMMARY.md`, which now sits
  beside this file.

The audit itself is reported in a new Section 4.4, `sec:discovery`, with Table 3
(`tab:clustering`) giving the correlation under all four labelings. Every value was
verified against the two summary JSONs at commit `51ff250` before transcription, with
the source paths in comments beside the table. No theorem statement was touched, so the
freeze state is unchanged.

The E-C oracle numbers in Table 2 come from the full 30 and 44 item samples and are not
to be mixed with the clustering run's oracle column, which excludes the pilot IDs and
runs on 25 and 39 items. On that smaller sample one of the four oracle split-half
intervals includes zero ($0.698$, $[-0.024, 0.951]$, Qwen first-half predictor), which
is a sample-size difference and not a disagreement.

## Reading discovery: what is known, and the repair to try first

The frozen run separates two questions that had been treated as one. Detecting that a
question admits more than one reading works reasonably: the Q lister returns multiple
interpretations on $80.0\%$ of Qwen Set A items and $51.3\%$ of Llama Set A items,
against Set B false-positive rates of $24.0\%$ and $22.5\%$. Recovering the right pair
of readings almost never works: all annotated readings are recovered on $1/25$ and
$1/39$ items. Assignment, given a usable list, is accurate, agreeing with oracle labels
on $90.0\%$ and $86.3\%$ of uniquely matched responses under Q. So the bottleneck is
the reading set, not the assignment step and not the estimator.

Q is preferred to QS wherever a variant is used. QS detected no more Set A ambiguity
while roughly doubling the Set B false-positive rate.

The repair candidate, recorded now so it is not rediscovered later: the failed pipeline
discovers readings from the *question*, through the lister. The natural alternative
discovers them from the *answers*, by semantically clustering the 32 samples, which is
semantic entropy's own machinery and the head-to-head comparison the reception pass
asked for on other grounds. The answer-string arm already in the run is the degenerate
form of that idea, one cluster per distinct normalized string, and the frozen run shows
why the degenerate form does not settle the question: on Llama its correlation is
$0.321$, comparable to Q, while its Set B observed categorical variance is $0.2396$
against the oracle arm's $0.0406$, so it manufactures apparent structure from ordinary
answer scatter. Entailment-based clustering is the intermediate case, and it is the
repair arm to run.

It rides along with the semantic-entropy baseline, which is why it moved onto the
submission path as item 4 below (Adam, 2026-08-25). Semantic entropy is computed by
clustering the sampled answers under bidirectional entailment and taking the entropy
over the cluster masses, so running that baseline is running answer-side clustering.
The same labels then give the entropy over cluster masses, which is the baseline the
reception pass asked for, and the between-cluster variance of the answer indicator,
which is the estimator under inferred labels. The rest is built already, since
`ec/clustering.py::estimate_from_labels` takes labels and calls the same
`decompose_conditionals` as the oracle arm, the split-half bootstrap is in place, and
the maximum-overlap agreement audit that produced the assignment figures applies
unchanged. No subject-model resampling is needed: the three 32-sample batches per item
are on disk, as the Q and QS arms already showed by reusing them.

The one genuinely new cost is the pairwise entailment budget over 32 samples per prompt,
which George sizes against his node, with the greedy agglomerative shortcut of the
semantic-entropy papers as the cheaper form. The clusterer, whether an NLI model or an
LLM prompted for entailment, is chosen and committed before the run, the way the lister
prompts were.

The numbers to beat are on the record. If entailment clustering holds a Set A
correlation whose interval clears zero while its Set B observed categorical variance
lands near the oracle arm's $0.0761$ and $0.0406$ rather than the answer-string arm's
$0.5680$ and $0.2396$, C5 is reclaimed and contribution 4 is restated in its stronger
form. If it lands near the string arm, the rescope stands and the limitations section is
better evidenced, having shown the failure persists under the obvious repair. Either way the
arm earns its place, which is what the rule at the top of this file asks of an
experiment. It stays off the gate list, so it cannot delay the submission.

## The experiment queue

Runs are cheap, about three minutes per arm on DGX-26, so the constraint throughout is
design time and George's attention, not compute. Item pools are not sized or reused to
save compute.

The order below was re-derived on 2026-08-25 when the paper paused. The previous ranking
came from what the submission needed, which is no longer the thing being optimized. Two
changes follow. E-A moves up from "if time allows" to a run worth making on its own
terms: it is designed, harnessed, preregistered, and unrun, and it is the only
experiment that tests the mechanism behind C2 in a deployed model rather than asserting
it as a theorem. And the preregistration becomes the critical path rather than a
checklist item, since it gates every publication-grade run and the exploratory runs'
recorded weakness was a protocol written alongside the implementation.

Unblocked, able to start before the preregistration freezes:

1. **Experiment D, reading discovery at scale.** One frozen Q lister call per item over
   the full 2,956-item two-reading AmbigQA pool plus an unambiguous control set, then an
   evaluation-only judge stage for the recall audit. No subject-model samples. This
   decides how the paper states the discovery result: broadly, or concentrated on the
   items where ambiguity is subtlest. The frozen run measured the confidence-screened
   slice, roughly 1 to 1.5% of the pool, selected for ambiguity subtle enough to leave
   the model near coin-flip confidence, which is close to adversarial for a lister. D is
   nearly free and runs ahead of everything. It reports detection and recovery
   separately, since the frozen run shows they come apart. Specified in
   `FOR_GEORGE_population_redesign.md`, beside this file. Its own decisions, listed
   there, are the control set's source and size, the judge model for the recall audit,
   and whether QS is carried; none of them waits on the E-B preregistration.
2. **E-A, the freeze test.** Promoted 2026-08-25 from "if time allows". It tests the
   mechanism behind C2 in a deployed model, which the paper currently asserts as a
   theorem about the collapse operator while making no claim about any real system, and
   it is the only queued run that addresses that gap. It is also unblocked: the
   preregistration, harness, reference paths, arithmetic control, mode-collapse
   detector, and power calculations are already fixed in `code/freeze_test_prereg.md`
   and `code/freeze_test.py`, with George's copy in the repo's `freeze_test/`. The
   registered prediction is the typed path $m + v/m = 0.52$ after one head against the
   frozen $0.50$. Either outcome is informative: frozen-path behavior gives the
   mechanism in a deployed model, and typed-path behavior locates collapse at the
   reporting interface instead, contradicting neither the theorems nor E-B.

Waiting on the preregistration freeze:

3. **Publication-grade E-B on the frozen slate**, serving C1, the paper's central
   negative. Its present base is two mid-size models at $n = 30$ and $44$, exploratory.
   The slate is Qwen3-32B, Llama-3-70B-Instruct, and one frontier API model as the
   stated kill-condition test, with an optional third family first to drop. The
   confidence band's fallback rule is fixed in the prereg before anything runs: the
   50-60% band caught 1.0-1.5% of the pool for mid-size models and may starve on a
   frontier model.
4. **E-C with oracle clustering on the same slate**, serving C4. Cheap next to E-B, and
   the estimator table replicated across the slate is the strongest form of that claim.
5. **Answer-side clustering at matched sample budget**, one run with two readouts,
   serving the novelty defense and the repair of C5 together. Cluster the sampled
   answers under bidirectional entailment, then report the entropy over cluster masses,
   which is the semantic-entropy baseline the reception pass asked for, beside the
   between-cluster variance of the answer indicator, which is the estimator under
   inferred labels. Both go in the Section 4.4 table next to the oracle, Q, and
   answer-string columns, with the Set B diagnostics beside them, since a Set A
   correlation means nothing without the false-structure control. The clusterer is
   committed before the run and the pairwise entailment budget is George's call.
6. **Reversed-order Qwen Set B control**, closing the stated caveat on the repeat-batch
   asymmetry. Already agreed and small.

Deferred, with the decision point named:

- **Experiment P, correlation preservation on a chosen sample.** Unscreened by
  confidence, sized by power; $n = 300$ separates oracle-level from collapsed-level
  correlations. Valuable, and under the adopted rescope it sharpens a result the paper
  already reports with its population named, so it can land in the appendix late or in
  a later version.
  Decide after D reports. The answer-side clustering arm moved out of here to item 4,
  since it shares its expensive step with the semantic-entropy baseline, which is on the
  submission path already; what stays deferred is P's larger unscreened population.
- **Third model family**, Gemma-3-27B or Mistral-Small class. First to drop under time
  pressure.

Freeze before the publication-grade runs start: the confidence band and its fallback,
the grading rule, the three predictions, the elicitation form, exclusion rules, seeds,
the entailment clusterer and its pairwise budget for item 4, and Experiment P's
population and $n$ if P runs. The exploratory runs' main weakness was a protocol written
alongside the implementation.

Operational notes carried into the runs. Every set is rescreened per model, since the
sets are per-model matched. The leak audit matters more at frontier scale, not less: a
more capable simulator volunteers answers more often, so more than the one drop in 45
should be budgeted for in the item counts, with both filtered and unfiltered summaries
kept in the record. For the frontier API model the budget is dominated by screening,
then the four E-B conditions, then the 32-sample E-C batches. George confirms which
Qwen3-32B release runs cleanly under vLLM on the node. Model and exact version are
recorded at run time, and every reported value derives from a committed machine-readable
summary with its source path recorded. Qwen3-8B is retired from the study but not
erased: its exploratory numbers stay as the record of the competence case, where
available value went entirely uncaptured, at minimum as an appendix note.

## Theory additions (held during the pause)

- The debiased estimator: the one-way variance-components correction for the plug-in's
  upward bias, more than a factor of two at the protocol's $n = 8$, with
  `code/estimator_check.py` extended to verify it. Without it the thresholded rule is
  not stateable, so it carries whatever remains of C5 and is desk work that does not
  wait on George.
- One general inequality, a value-of-information or excess-loss lower bound in terms of
  the between-reading variance. Per `LITERATURE_SWEEP_P2.md` any such statement cites
  the known identity (Chen-Waggoner, Frankel-Kamenica, Murphy) and claims only the
  typed-belief instantiation. Both reception passes name this as the addition that would
  raise the contribution score.
- Held, not adopted: the elicitation possibility theorem complementing Bengs (a
  conjecture, `statement-check` due) and the sequential-asking example. `STATE.md`
  records both.

Question selection is out of the main document, settled 2026-08-25. The design
separation and expected information gain have no home in the four-part structure; the
cross-category value returns as one remark in the estimator section if it earns the
space.

## Timing (the paper side, held)

The referee pass covering the 2026-08-24 fixes, the DGX-26 rewrite, the front-matter
application, and now the rescope is the gate before the draft is shown to anyone, and it
runs in a fresh session. thm:freeze, thm:factor, prop:reveal, and prop:consistent are
live until it clears. The rescope is applied, so one pass now covers everything.

Compression to nine pages is a major change and a fresh referee pass follows it, so
allow about a week from final numbers to submission. Working back from September 25,
experimental values need to be settled around September 12 to 15, which puts the prereg
freeze inside this week. The theory additions and the debiased estimator run in parallel
and are not blocked on any run.

`STATE.md` holds the open items, the freeze state, and the verification ledger.

## Where things live

The project folder holds the manuscript in `LLM UQ latex/`, the verification scripts in
`code/`, the retired paper 1 in `_TU arxiv/`, and the planning set in
`document rules and plans/`: this file, `STATE.md`, `REFEREE.md`,
`MOCK_REVIEWS_redesign_P1.md`, `LITERATURE.md`, `LITERATURE_SWEEP_P2.md`,
`EC_HYPOTHESIS_SUMMARY.md`, and `FOR_GEORGE_population_redesign.md`. George's
experimental records are not mirrored here: `experiments-repo.webloc` at the project
root is the link, and the repo is cloned fresh when its records are needed, which keeps
one copy of every number. The project is not under version control, per section 11 of
the conventions.
