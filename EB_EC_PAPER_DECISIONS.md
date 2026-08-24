# E-B/E-C decisions for the paper

Author decision record from George Adamopoulos.

Revision 2, 2026-08-24: adds the split-half re-estimate, the null-control
investigation, and the within-band framing of the confidence comparison, in
response to Adam's review of 2026-08-24. Sections 1–3 are unchanged in
substance; sections 4–6 are new.

## Status

The three questions raised about incorporating E-B and E-C into the paper are
resolved, and the three follow-up questions from the review are answered:

1. E-C uses oracle-style, fixed-reading clarification gain.
2. The paper will disclose the post-answer confidence deviation, its
   motivation, and the restriction it places on what the confidence comparison
   can claim.
3. The complete E-C protocol, raw results, and machine-readable summaries are
   committed (`1311f86`), together with the figure set (`9054bf6`) and the
   split-half analysis (`543b38f`).
4. The E-C correlation has been re-estimated split-half. It survives.
5. The Set B null-control asymmetry has been investigated. The order-drift
   mechanism is not supported by the data.
6. The confidence comparison will be reported as a within-band comparison.

## 1. Meaning of E-C clarification gain

E-C does **not** use the model's self-ask condition to define gain. It measures
competence-free, oracle-style clarification value.

For each ambiguous Set A item and each of its two annotated readings:

```text
baseline accuracy   = accuracy under the original ambiguous question
clarified accuracy  = accuracy under that reading's annotated fixed rewrite
reading-level gain  = clarified accuracy - baseline accuracy
```

The item-level E-C gain is the uniform average of the two reading-level gains.
Each prompt receives 32 samples at temperature 1.0.

This is the intended theoretical quantity because it measures the value made
available by resolving the interpretation without conflating that value with
the model's ability to formulate a good question, understand the reply, and
use it correctly.

### Paper wording

> E-C clarification gain is the improvement from replacing the ambiguous
> prompt with an annotated fixed-reading rewrite, averaged uniformly over the
> two possible intended readings. It therefore estimates the available value
> of clarification independently of the model's competence in formulating or
> using a clarifying question. Because it requires the dataset's annotated
> rewrites, it is an oracle quantity: an upper bound on what clarification
> makes available, not a diagnostic computable at inference time.

### Qwen3-8B competence limitation

The Qwen results illustrate the distinction between available value and the
ability to capture it:

- E-B oracle clarification gain: **+13.3 percentage points**.
- E-B self-ask gain: **0 percentage points**.
- E-C repeated-sampling oracle-style gain: **+10.1 percentage points**.

The E-B oracle and E-C values are related but are not identical estimands, and
must not be differenced. They differ in three ways:

1. **Readings.** E-B evaluated one randomly selected intended reading per
   question; E-C evaluated both and averaged the two reading-level gains.
2. **Sampling.** E-B used a single response per condition at temperature 0;
   E-C used 32 samples per prompt at temperature 1.0, so its accuracies are
   fractions rather than 0/1 outcomes.
3. **Baseline.** E-B's `answer_now` is the single answer generated during
   confidence screening — the answer whose confidence the model rated. E-C
   does not reuse it; its baseline is a fresh 32-sample batch drawn from the
   ambiguous prompt at temperature 1.0.

### Two-stage-section wording

> For Qwen3-8B, oracle-style clarification had positive value, while
> self-asking captured none of it, illustrating a competence gap between the
> value of information and the ability to obtain and use it.

## 2. Confidence-elicitation deviation

The original written design proposed eliciting confidence from the bare
question before the model answered. That procedure was tested first but
collapsed: the model reported approximately 85–95% confidence on nearly every
question and produced no usable matches in the intended 50–60% band.

The executed E-B experiment therefore used post-answer confidence:

1. The model generated a direct answer.
2. In a separate stateless call, the model rated its confidence that this
   proposed answer was correct.
3. Items reporting confidence from 50% through 60%, inclusive, were retained.
4. The screening answer was reused as the answer-now response because it was
   the answer whose confidence had been rated.

E-C did not elicit confidence again. It inherited the matched E-B samples and
their post-answer confidence measurements.

### Methods wording

> Items were matched using post-answer verbalized confidence: the model first
> generated an answer and then rated its confidence in that answer. We adopted
> this procedure because blind question-only confidence collapsed near 85–95%
> and yielded no usable items in the target 50–60% band.

This is a substantive protocol deviation and will be stated explicitly. The
reported experiments should be described as exploratory rather than as a
pristine execution of a frozen preregistration.

### What the deviation costs, and the claims it restricts

Filtering post-answer integer confidence into a 10-point band leaves the
variable with almost no variance. In the E-C records, **every Qwen3-8B item is
either 50 or 60** (50 items at 50%, 10 at 60%); Llama-3-70B takes four values
across 89 items, overwhelmingly the same two.

Two reported results therefore compare against a predictor the design
deliberately flattened, and must be restated:

- **E-C prediction 4** ("between-reading variance beats scalar confidence",
  advantage 0.937 [0.353, 1.251] for Qwen) is a **within-band** comparison. The
  claim that between-reading variance outperforms *undifferentiated answer
  variance* is the fair half of prediction 4 — both predictors retain full
  range — and will carry the claim on its own.
- **E-B prediction 2** (confidence vs. realized gain, r = +0.079 and −0.100) is
  not the evidence that confidence cannot decide whether to ask. The evidence
  is the **between-set contrast**: matched confidence, different gain. The
  within-sample correlation will be reported as a within-band observation, not
  led with, because the range was restricted by our own design.

## 3. Artifact and reproducibility record

All artifacts are committed. Values are derived from the machine-readable
summaries; no number is transcribed from a prose document.

### Protocol and human-readable reports

- `ec_protocol.md`
- `EC_EXPERIMENT_RESULTS.md`
- `EB_EXPERIMENT_SUMMARY.md`

`EC_HYPOTHESIS_SUMMARY.md` is **deliberately not committed**. It is an internal
interpretation memo whose bottom line ("strongly support the hypothesis") is
stronger than sections 4 and 6 below justify. It will be committed only after
it is revised to carry the same qualifications as the paper.

### E-C artifacts

- `scan_results/ec_qwen3_8b_results.jsonl` + `.summary.json` + `_run.log`
- `scan_results/ec_llama3_70b_results.jsonl` + `.summary.json` + `_run.log`

### E-B artifacts

- `scan_results/full_eb_qwen3_8b_results.jsonl` + `_summary.json`
- `scan_results/full_eb_llama3_70b_results.jsonl` + `_summary.json`
- `scan_results/full_eb_llama3_70b_summary_leak_filtered.json` — **the
  reporting summary for Llama**
- `scan_results/full_eb_*_setb_screen.jsonl` — backs the match-rate table
- `scan_results/seta_fullscan_*.jsonl` + `.summary.json` — backs the Set A
  exhaustion claim

### Figures

`figures/make_figures.py` regenerates all five figures from the artifacts
above and prints the source of each. `figures/README.md` maps every figure to
the claim it carries and the caveat its caption must state.

The raw files contain 60 Qwen records and 89 leak-filtered Llama records. The
maximum recorded variance-decomposition identity error is 0.0 for both runs —
**this will not be reported.** It checks that two algebraic forms of the same
identity agree in floating point, which is a unit test on the estimator, not
evidence about the experiment, and in a methods section it would read as the
latter.

### How the incomplete copy was produced

The handoff archive carried a git repository whose single commit predates the
DGX work. `EB_EXPERIMENT_SUMMARY.md`, `EC_EXPERIMENT_RESULTS.md`, `ec/`,
`ec_protocol.md`, `scan_results/`, and `tests/test_ec.py` were untracked
working-tree files; `experiments-run-with-results.md` was tracked in its
pre-supersession form. Any copy made *through git* — clone, push, or
`git archive` — therefore drops exactly the files that went missing and
retains the stale Vulcan document looking current. Copying the working tree
directly, as `TRANSFER_MANIFEST.md` instructed, does not. Now that everything
is committed, a clone reproduces the full set and the failure cannot recur by
that path.

## 4. Split-half re-estimate of the E-C correlation

The published E-C correlation estimates the predictor (between-reading
variance) and the outcome (realized gain) from the same 32 samples per
fixed-reading prompt, so shared sampling noise inflates it. This has been
re-estimated from disjoint halves of the existing samples: the predictor from
16 draws per prompt, the outcome from the other 16. No new inference was
required. Implementation: `ec/split_half.py`.

Both half-assignments are reported, since neither half is privileged. The
baseline is not split: it comes from the ambiguous prompt, which never enters
the variance estimate.

| Model | Same samples (published) | Variance ← 1st half | Variance ← 2nd half |
|---|---|---|---|
| Qwen3-8B (n=30) | +0.790 [0.349, 0.950] | +0.684 [0.103, 0.934] | +0.829 [0.499, 0.946] |
| Llama-3-70B (n=44) | +0.613 [0.375, 0.809] | +0.582 [0.334, 0.787] | +0.610 [0.370, 0.807] |

All four split estimates are positive with 95% intervals excluding zero. Llama
is barely attenuated, as expected at n=44; Qwen swings more, as expected at
n=30 with half the draws.

**Decision.** The split-half estimates become the reported correlation. The
same-sample estimate moves to an appendix, described as the inflated
within-sample version. Both half-assignments are reported rather than one
chosen. Figure 2 shows all three.

## 5. The Set B null control

Qwen's Set B repeat-batch control came in at +2.4pp on identical prompts, 8
items up against 1 down (exact sign-flip p ≈ 0.012 one-sided). Llama's control
is clean (3 up, 1 down, p = 0.31).

The proposed mechanism — a monotone drift favouring later batches, which would
inflate Set A because its clarified batches always run after the ambiguous one
— is **not supported**. Within Set A each item runs ambiguous → reading_a →
reading_b, so drift predicts batch 3 > batch 2. The observed direction is the
opposite:

| Model | batch #2 (reading_a) | batch #3 (reading_b) | difference |
|---|---|---|---|
| Qwen3-8B | 22.1% | 5.6% | −16.5pp |
| Llama-3-70B | 37.6% | 26.1% | −11.5pp |

This comparison is confounded — the two readings are different questions and
AmbigQA's ordering is not random — so it does not measure position cleanly.
What it rules out is a *positive* monotone position effect large enough to
account for a quarter of Qwen's +10.05pp. Two further checks are also
negative: drift does not correlate with the original batch's accuracy
(r = +0.05), and it favours the first half of the run only mildly (+3.8pp vs
+1.0pp), so global server warm-up does not explain it either.

**Decision.** Treat the asymmetry as a probable chance fluctuation surfaced by
one of several checks, and say so. Re-run the Qwen Set B control with the
batch order reversed to close the question — framed as ruling out a
possibility, not as measuring a known bias. Llama's numbers are unaffected
either way.

**Separately:** the −16.5pp / −11.5pp gap between the two readings is a
finding in its own right. E-C averages the readings uniformly, which assumes
they are exchangeable; they evidently are not. This should be checked before
the uniform average is defended in print.

## 6. Smaller items

### The Llama self-ask leak

One of 45 Llama Set A self-ask replies leaked the intended answer (2.2%),
above the "one or two percent" threshold at which
`experiment-ask-protocol.md` calls for inspecting the rewrite set. The
inspection was done. Item `9087726812198390660`:

- Question: *"Who sang only love can break your heart?"*
- Model's clarifying question: *"Was the song a hit in the 1970s?"*
- Simulator reply: *"No, I meant the 1990s song by Saint Etienne."*

The simulator volunteered the answer from its own knowledge; the rewrite never
contained it. AmbigQA disambiguates by adding a qualifier rather than by
naming the answer, which sibling items confirm ("How many court of appeals
**divisions**…", "When did **season 2** of the Expanse start?"). The rewrite
set is therefore not contaminated and the threshold's underlying concern does
not apply.

**Decision.** Dropping the item is correct; report both the unfiltered and the
leak-filtered summaries, with the leak-filtered one as the Llama result. Note
that a more capable simulator will leak more often, so the audit must remain
and the rate should be expected to rise with frontier models.

**Gap to close:** E-B result records do not store the rewrite text, so the
leak audit is not re-checkable from the artifacts alone. Add the rewrite to
the E-B record before the next run.

### Presentation of Qwen's Set B result

Qwen's Set B self-ask gain is +10.0pp [−3.3, +23.3] against a Set A self-ask
gain of exactly 0. The point estimate inverts prediction 1 within that model.

It should be met directly, with the reason it is benign: **Qwen scored 0/30 on
Set A in both answer-now and self-ask.** The zero gain is a floor, not a null.
The same model reaches 4/30 under oracle disambiguation, so the value is real
and reachable; it simply never converts a self-asked exchange into a correct
Set A answer. A set pinned at zero accuracy cannot furnish a fair within-model
test of prediction 1.

**Decision.** Llama carries prediction 1; Qwen is presented as the competence
case framed in section 1. The Set B point estimate is reported with its
interval noted, together with the floor explanation. The free-choice behaviour
supports the split: Llama asks on 40.9% of Set A against 8.9% of Set B, while
Qwen asks at 36.7% and 40.0%, showing no behavioural separation.

**One further number to meet deliberately:** Qwen's free-choice Set B gain is
+20.0pp [+3.3, +36.7] — an interval excluding zero, and the largest positive
Qwen effect anywhere in the study. Nothing should be claimed from it at n=30
among many reported quantities, but a referee reading the free-choice row will
find it, so it should be addressed in the same paragraph rather than
discovered.

## Commitment before manuscript transcription

Before adding an E-B or E-C number to the `.tex` source:

- derive each table or figure value from the machine-readable summary;
- record the relative source path in the figure-generation code or caption
  notes;
- use the leak-filtered Llama E-B and E-C samples;
- report the split-half E-C correlation as the primary estimate;
- preserve the distinction between oracle-style value and realized self-ask
  competence;
- state the post-answer confidence procedure wherever confidence matching is
  described, and mark the confidence comparison as within-band;
- omit the decomposition identity error.

## Reply to the collaborator

> Agreed on all three original points, and thank you for the three follow-ups —
> two of them turned out to be answerable from the existing records.
>
> **The missing files.** The handoff archive carried a git repository whose one
> commit predates the DGX work, and both summaries, the whole `ec/` package,
> and `scan_results/` were untracked working-tree files. Any copy made through
> git drops exactly those and keeps `experiments-run-with-results.md` in its
> pre-banner form, which is precisely the symptom you saw. Everything is now
> committed and pushed to `main` on
> `github.com/george-adams1/latent_uncertainty_estimation_experiments` — please
> pull rather than working from the copy you have, since a clone now reproduces
> the full set including the raw records and the summary JSONs.
>
> **Please delete the Qwen2.5-72B-Instruct note rather than updating it.** That
> model is not in the final experiments at all. It appears only in the Vulcan
> development runs at n=20 per set, and the superseded document you were reading
> states in as many words that it is "the clearest confirmation of prediction 1
> across both runs" at +20pp oracle, +20pp self-ask, +0pp Set B. The executed
> experiments are Qwen3-8B and Llama-3-70B only; no result in the paper should
> cite a third model. The repo copy of that document now carries a supersession
> banner at the top and has moved to `docs/provenance/`.
>
> **On the three caveats.** All three now have their own sections in the
> decisions record rather than living only in caption notes: split-half is §4,
> the null control §5, and the confidence restriction a new subsection of §2.
> `figures/README.md` still carries them as caption guidance, and Figure 2 now
> reports the split-half correlation alongside the same-sample one.
>
> **Split-half.** Done, no new inference needed. Predictor from 16 draws,
> outcome from the disjoint 16, both assignments: Qwen +0.684 [0.103, 0.934]
> and +0.829 [0.499, 0.946]; Llama +0.582 [0.334, 0.787] and +0.610 [0.370,
> 0.807]. All four exclude zero. I have made split-half the reported estimate
> and moved the same-sample figure to an appendix, exactly as you proposed.
>
> **The null control.** The asymmetry itself is real in that sample — an exact
> sign-flip test over the nine items that moved gives p ≈ 0.012 one-sided, which
> agrees with your interval. What I cannot support is the inference from it to
> Set A. Within Set A the batches run ambiguous → reading_a → reading_b, so an
> upward drift predicts batch 3 > batch 2; the observed difference is −16.5pp
> for Qwen and −11.5pp for Llama. Drift also does not track the original
> batch's accuracy (r = +0.05) and is only mildly concentrated early in the run.
> So the Set B result stands and the quarter-of-+10.05pp correction does not
> follow from it. I read the asymmetry as a probable fluctuation, will say so,
> and will still run the reversed-order Qwen control to close it. Note the
> reading_a/reading_b gap is its own finding: E-C averages the two readings
> uniformly, and they are not exchangeable.
>
> **Confidence.** Agreed, within-band, and I would go a step further: for Qwen
> the variable takes two values, so I am letting the comparison against
> undifferentiated answer variance carry prediction 4 and reporting the
> confidence comparison as a within-band observation only. `EC_HYPOTHESIS_SUMMARY.md`
> stays out of the repo until it carries the same qualification.
>
> **The leak.** One clarification on the protocol: it mandates the drop
> regardless of rate — "drop any item that hits, and report the drop rate" —
> and what the one-or-two-percent threshold triggers is inspection of the
> *rewrite set*, on the theory that a high rate means the rewrites are
> contaminated. So I did the inspection. The simulator volunteered "Saint
> Etienne" from its own knowledge; the rewrite never contained it, and AmbigQA
> disambiguates by qualifier rather than by answer. The rewrite set is clean,
> so the threshold's concern does not apply and the drop is protocol-compliant
> rather than a judgment call. I will report both summaries. The audit stays,
> and I expect the rate to rise with more capable simulators.
>
> **Identity error.** Agreed, dropping it — it is a unit test on the estimator.
>
> **Presentation.** Agreed with your split, with one addition: Qwen scored 0/30
> on Set A in both answer-now and self-ask, so its zero gain is a floor rather
> than a null, and a set pinned at zero cannot give a fair within-model test of
> prediction 1. That is the sentence I will use. One number to meet
> deliberately in the same paragraph: Qwen's free-choice Set B gain is +20.0pp
> [+3.3, +36.7], the largest positive Qwen effect in the study.
