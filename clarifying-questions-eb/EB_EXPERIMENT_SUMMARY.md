# E-B experiment summary

This document records the E-B matched-confidence ask-or-answer experiment
executed on 2026-08-19. It describes the scientific experiment and its final
results, rather than the infrastructure used to run it.

## Research question

E-B tests whether a language model's scalar confidence is sufficient to decide
when asking a clarifying question will help.

The motivating distinction is between two reasons for uncertainty:

- **Set A: ambiguity.** The question has two valid interpretations. Asking the
  user which interpretation they intend can resolve the uncertainty.
- **Set B: ordinary lack of knowledge.** The question has one interpretation,
  but the fact is difficult. Asking the user to clarify should not reveal the
  missing fact.

The two sets are restricted to the same verbalized-confidence band. If they
have similar confidence but different gains from clarification, confidence
alone cannot determine whether asking is valuable.

## Models evaluated

The complete experiment was run independently for:

- `Qwen/Qwen3-8B`
- `meta-llama/Meta-Llama-3-70B-Instruct`

The models were not pooled. Each model defines its own confidence-filtered
sample and is analyzed separately.

## Datasets and eligibility

### Set A: ambiguous questions

Set A came from `sewon/ambig_qa`, configuration `light`, using both the train
and validation splits. An item was eligible only when its annotation supplied
exactly two disambiguated questions, each with a nonempty answer-alias list.

This restriction produced 2,956 eligible candidates:

- 587 from validation
- 2,369 from train

All 2,956 candidates were screened for both models. This exhausts the Set A
pool under the current exactly-two-readings rule.

### Set B: difficult unambiguous questions

Set B came from `mandarjoshi/trivia_qa`, configuration
`unfiltered.nocontext`. The final matched samples were drawn from its
validation split, which contains 11,313 candidates. TriviaQA supplies answer
aliases compatible with the mechanical grading used for Set A.

Set B screening stopped once enough confidence-band matches had been found to
match the number of Set A items. Therefore, Set B was not exhausted.

## Confidence matching

For every candidate, the model first produced a short answer and then rated
its confidence that this proposed answer was correct. Confidence was requested
as a single integer percentage.

Only candidates with confidence from **50% through 60%, inclusive**, were
retained. The answer produced during screening became the `answer-now` response
because it was the answer whose confidence the model had rated.

The resulting screening coverage was:

| Quantity | Qwen3-8B | Llama-3-70B |
|---|---:|---:|
| Eligible Set A candidates screened | 2,956 | 2,956 |
| Parsed Set A confidence values | 2,956 | 2,956 |
| Set A confidence-band matches | 30 | 45 |
| Set A match rate | 1.01% | 1.52% |
| Set B validation candidates screened | 4,096 | 8,192 |
| Set B confidence-band matches found | 33 | 62 |
| Set B matches selected | 30 | 45 |

For each selected Set A item, one of its two readings was fixed uniformly at
random with seed 0 before the experimental conditions were evaluated.

## Experimental conditions

Every selected item was evaluated through stateless condition calls. Calls
did not share conversation history across conditions.

### 1. Answer now

The model answered the original question without clarification. As described
above, this was the answer generated during confidence screening.

### 2. Oracle clarification

For Set A, the ambiguous question was replaced directly with the rewrite for
the preselected intended reading. This removes the model's ability to write a
good clarifying question from the measurement and estimates the maximum gain
available from disambiguation.

Set B has no oracle-clarification condition because it has only one reading.

### 3. Self-ask

The model wrote one clarifying question. A simulated user answered it using
the intended reading for Set A. The model then gave a final answer using the
original question, its clarifying question, and the simulated reply.

For Set B, the simulator indicated that the question was meant exactly as
asked and contained no hidden ambiguity. This tests whether an unnecessary
clarification turn improves a hard factual answer.

The simulator was never given the intended answer. Its replies were audited
afterward for accidental answer leakage.

### 4. Free choice

The model chose whether to ask a clarifying question or answer immediately.
If it asked, the same simulated-user exchange was performed. This condition
measures whether the model's asking behavior tracks ambiguity or the realized
benefit of asking.

## Typed diagnostic

For every selected item, the model generated eight independent answers at
temperature 1.0. Samples were mechanically clustered by answer-alias matches:

- Set A had one cluster for each of its two readings.
- Set B had one known-answer cluster.
- Samples matching no known alias were retained in an `other` cluster.

Two summaries of this distribution were tested as predictors of clarification
gain:

- entropy across the observed answer clusters;
- the mass of the second-largest cluster.

This is a typed diagnostic embedded in E-B. It is related to the proposed E-C
variance-estimator experiment, but it is not a complete standalone E-C run.

## Grading and outcome definitions

Answers were graded mechanically using normalized answer-alias containment.
Normalization lowercased text and removed punctuation and articles.

For Set A, the answer was compared with the aliases of the preselected
intended reading. An answer containing aliases from both readings was labeled
`hedged`. Strict headline accuracy scores `correct` as 1 and both `wrong` and
`hedged` as 0.

The primary gains were within-item differences in strict correctness:

- **Set A oracle gain:** oracle clarification minus answer now;
- **Set A self-ask gain:** self-ask minus answer now;
- **Set B self-ask gain:** self-ask minus answer now.

## Final sample and leak handling

Qwen produced 30 Set A and 30 Set B experimental records. No Set A simulator
reply leaked an intended answer.

Llama initially produced 45 Set A and 45 Set B records. One Set A self-ask
simulator reply explicitly contained the intended answer, `Saint Etienne`.
Item `9087726812198390660` was removed from the final analysis according to the
leak protocol. The final Llama analysis therefore contains 44 Set A and 45 Set
B items. No Llama free-choice reply leaked an answer.

## Headline results

### Accuracy by condition

Percentages in parentheses are 95% Wilson score confidence intervals.

| Model and condition | Accuracy |
|---|---:|
| Qwen Set A: answer now | 0.0% (0.0%, 11.4%) |
| Qwen Set A: oracle clarification | 13.3% (5.3%, 29.7%) |
| Qwen Set A: self-ask | 0.0% (0.0%, 11.4%) |
| Qwen Set B: answer now | 20.0% (9.5%, 37.3%) |
| Qwen Set B: self-ask | 30.0% (16.7%, 47.9%) |
| Llama Set A: answer now | 15.9% (7.9%, 29.4%) |
| Llama Set A: oracle clarification | 31.8% (20.0%, 46.6%) |
| Llama Set A: self-ask | 25.0% (14.6%, 39.4%) |
| Llama Set B: answer now | 51.1% (37.0%, 65.0%) |
| Llama Set B: self-ask | 53.3% (39.1%, 67.1%) |

### Gain from clarification

Percentages are percentage-point changes. Intervals are 95% item-level
percentile-bootstrap intervals with 10,000 resamples, paired within item.

| Gain | Qwen3-8B | Llama-3-70B |
|---|---:|---:|
| Set A oracle gain | +13.3 pp (+3.3, +26.7) | +15.9 pp (+4.5, +29.5) |
| Set A self-ask gain | 0.0 pp (0.0, 0.0) | +9.1 pp (+2.3, +18.2) |
| Set B self-ask gain | +10.0 pp (-3.3, +23.3) | +2.2 pp (-6.7, +11.1) |

Qwen had measurable oracle value on ambiguous questions but captured none of
it through its own clarifying exchange. Its observed Set B self-ask gain was
positive, but the interval includes zero.

Llama showed the intended qualitative separation: self-asking improved Set A
by 9.1 points and Set B by only 2.2 points. The Set A interval excludes zero;
the Set B interval does not.

### Confidence and typed-diagnostic correlations

Correlations pool Set A and Set B within a model. The 95% intervals use 10,000
bootstrap resamples, stratified by set.

| Correlation | Qwen3-8B | Llama-3-70B |
|---|---:|---:|
| Confidence vs. realized self-ask gain | +0.079 (-0.127, +0.363) | -0.100 (-0.280, +0.122) |
| Diagnostic entropy vs. gain | +0.109 (-0.212, +0.452) | +0.112 (-0.219, +0.434) |
| Second-largest cluster vs. gain | +0.234 (-0.181, +0.616) | +0.101 (-0.195, +0.413) |

Confidence was close to uncorrelated with realized gain in both models, as
predicted. Both typed-diagnostic correlations were positive, also in the
predicted direction, but their intervals are wide and include zero.

### Free-choice behavior

| Metric | Qwen3-8B | Llama-3-70B |
|---|---:|---:|
| Set A asking rate | 36.7% (21.9%, 54.5%) | 40.9% (27.7%, 55.6%) |
| Set B asking rate | 40.0% (24.6%, 57.7%) | 8.9% (3.5%, 20.7%) |
| Set A free-choice gain | +6.7 pp (0.0, +16.7) | +4.5 pp (-4.5, +13.6) |
| Set B free-choice gain | +20.0 pp (+3.3, +36.7) | -2.2 pp (-6.7, 0.0) |

Llama asked much more often on ambiguous Set A than on unambiguous Set B.
Qwen asked at similar rates on both sets and therefore did not display this
behavioral separation.

## Statistical uncertainty

The reported intervals were added after the experimental records had been
generated; no new model inference was required.

- Binomial accuracies, hedge rates, and asking rates use 95% Wilson score
  intervals.
- Gains use 10,000 deterministic percentile-bootstrap resamples, paired
  within item.
- Pooled correlations use 10,000 deterministic bootstrap resamples,
  stratified by Set A and Set B.
- Bootstrap seed: 0.

The sample sizes remain small. The gain results for Llama are more informative
than the correlations, whose intervals are broad for both models.

## Deviations from the originally described protocol

This was the complete executed E-B experiment, but it was not a pristine
confirmatory run of a frozen preregistration.

1. **Post-answer confidence was used.** The original design requested a blind
   confidence score before answering. Preliminary runs showed that capable
   models collapsed toward approximately 85-95% on nearly every question,
   yielding no usable 50-60% matches. The executed experiment therefore asked
   the model to rate a specific answer it had just produced.
2. **The diagnostic used eight samples rather than ten.** Eight was the
   implemented and executed value for both models.
3. **The preregistration remains a draft.** It was written alongside the
   implementation and was not timestamp-frozen before these runs.
4. **Only exactly-two-reading AmbigQA items were included.** Questions with
   three or more annotated readings were excluded rather than generalized to
   a multi-reading protocol.
5. **Strict mechanical grading was used.** There was no semantic judge; valid
   paraphrases missed by the supplied alias lists could be scored as wrong.

These choices are part of the estimand actually measured and should not be
silently changed when reproducing these numbers.

## Interpretation

The strongest result is from Llama. Questions matched to the same 50-60%
confidence band behaved differently depending on the source of uncertainty:
clarification helped genuinely ambiguous questions, while it had little
measured benefit on hard unambiguous trivia. At the same time, confidence did
not predict the item-level gain.

Qwen supplies a different but useful failure case. Oracle disambiguation had
positive value, showing that information was available, but the model failed
to extract that value through its own self-ask interaction. This separates the
value of clarification from the competence required to formulate and use a
clarifying exchange.

The typed diagnostic was directionally consistent with the hypothesis, but
this experiment does not establish that it reliably predicts gain. A larger
or dedicated E-C experiment would be needed for that claim.

## What data remain available

Set A is exhausted under the present eligibility and confidence rules: every
one of the 2,956 eligible two-reading candidates was screened for each model.
There are no additional independent Set A matches to add without changing the
design.

Set B is not exhausted:

- Qwen left 7,217 validation candidates unscanned and had 3 already-screened
  matches beyond the 30 selected.
- Llama left 3,121 validation candidates unscanned and had 17
  already-screened matches beyond the 45 selected.
- TriviaQA also has approximately 87,622 training questions.

Additional Set B records could tighten the estimate of the unambiguous
self-ask effect, but they would not solve the primary Set A sample-size
bottleneck.

Possible extensions, each requiring an explicitly documented analysis or
protocol change, are:

1. Evaluate both intended readings for each matched Set A question. This could
   produce up to 60 Qwen and 90 Llama item-reading observations, but the two
   observations from one question are dependent and require question-clustered
   inference.
2. Generalize the harness to AmbigQA items with three or more readings.
3. Broaden the confidence band as a sensitivity analysis.
4. Add another ambiguity dataset.
5. Run a dedicated E-C variance-estimator experiment with more diagnostic
   samples and independent repetitions.

## Scope relative to E-A and E-C

This document covers E-B only.

- E-A is the separate synthetic freeze test comparing flat and typed belief
  representations.
- E-C is the proposed standalone variance-estimator experiment.

Neither E-A nor a complete standalone E-C experiment was run as part of the
work summarized here.

## Result artifacts

- [Qwen full Set A scan](scan_results/seta_fullscan_qwen3_8b.jsonl)
- [Qwen Set A scan summary](scan_results/seta_fullscan_qwen3_8b.jsonl.summary.json)
- [Qwen experimental records](scan_results/full_eb_qwen3_8b_results.jsonl)
- [Qwen Set B screening records](scan_results/full_eb_qwen3_8b_setb_screen.jsonl)
- [Qwen final summary](scan_results/full_eb_qwen3_8b_summary.json)
- [Llama full Set A scan](scan_results/seta_fullscan_llama3_70b.jsonl)
- [Llama Set A scan summary](scan_results/seta_fullscan_llama3_70b.jsonl.summary.json)
- [Llama experimental records](scan_results/full_eb_llama3_70b_results.jsonl)
- [Llama Set B screening records](scan_results/full_eb_llama3_70b_setb_screen.jsonl)
- [Llama unfiltered summary](scan_results/full_eb_llama3_70b_summary.json)
- [Llama leak-filtered final summary](scan_results/full_eb_llama3_70b_summary_leak_filtered.json)

The leak-filtered Llama summary is the final Llama result and should be used
for reporting.
