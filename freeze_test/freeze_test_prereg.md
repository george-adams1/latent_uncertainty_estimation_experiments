# Pre-registration: E-A, the freeze test

For `Clarifying Questions as Experimental Design`, testing Theorem 1 of
`Collapsing Uncertainty` on a deployed model. Written before any model has been
called. `paper2_plan.md` requires this frozen before the first run.

Implementation: `code/freeze_test.py`. Everything below is asserted by
`python3 freeze_test.py --selfcheck`, which reproduces every number here, and is
exercised end to end by `--dry-run`.

## Claim under test

A coin is drawn once, uniformly, from two coins with biases $(0.5-d, 0.5+d)$,
then flipped repeatedly. A reasoner that keeps the model channel updates its
forecast along the typed path. A reasoner that has collapsed to a single
confidence number is frozen at its prior forecast of 0.5 and cannot use the
history at all.

## Three reference paths, not two

Scored against `(d, n, s)`, a history of $s$ heads in $n$ flips:

| path | formula | behavior |
|---|---|---|
| typed | posterior over which coin, then marginalize | bounded in $(0.5-d,\,0.5+d)$, **saturates** |
| flat | $0.5$ | the collapse prediction, frozen |
| laplace | $(s+1)/(n+2)$ | frequency counting, **unbounded** |

The paper's framing is typed-or-frozen. Frequency counting is a third mode a
deployed model can plausibly be in, and a two-way design misattributes it. It is
registered as a scored outcome, not as a post-hoc explanation.

## Grid

Ten cells in the sampled channel, three more in the verbalized channel only.

| family | cells | role |
|---|---|---|
| saturation | $d{=}0.20$, $n \in \{10,20\}$, $s \in \{0,n\}$ | the decisive three-way test |
| extreme | $d{=}0.40$, $n{=}10$, $s \in \{3,7\}$ | the opposite geometry: typed extreme, counting moderate |
| paper | $d{=}0.30$, $n{=}1$, $s \in \{0,1\}$ | the paper's own cell; typed-vs-flat only |
| null | $d \in \{0.20, 0.40\}$, $n{=}10$, $s{=}5$ | harness control and mode-collapse detector |
| slope | $d \in \{0.05,0.10,0.30\}$, $n{=}1$, $s{=}1$ | verbalized only; the paper's E5 sweep |

**Why not the paper's cells.** Required samples to separate two paths at 80%
power, 5% two-sided:

| cell | typed | flat | laplace | $k$ typed vs flat | $k$ typed vs laplace |
|---|---|---|---|---|---|
| $d{=}0.05$, one head | 0.505 | 0.500 | 0.667 | **78,487** | 70 |
| $d{=}0.10$, one head | 0.520 | 0.500 | 0.667 | **4,904** | 85 |
| $d{=}0.30$, one head | 0.680 | 0.500 | 0.667 | 59 | **9,750** |
| $d{=}0.20$, $n{=}20$, $s{=}20$ | 0.700 | 0.500 | 0.955 | 47 | 10 |
| $d{=}0.40$, $n{=}10$, $s{=}7$ | 0.900 | 0.500 | 0.667 | 10 | 26 |

Two of the paper's three cells are below the noise floor of the primary channel.
The third looks like the good cell, because it separates typed from flat at
$k=59$, but it sits on top of the counting path and would score a counting model
as typed. The saturation cells separate all three at once, because the typed
path has a ceiling at the extreme bias and the counting path does not.

Within-history arrangement of heads is fixed by seed 20260810 rather than
blocked, so recency is not systematically confounded with $s$. For $s \in \{0,n\}$
there is one arrangement and the seed does nothing.

## Elicitation

**Primary: sampled-answer frequency**, $k = 120$ independent calls per cell per
condition, temperature 1, one token each; $\hat p = \#H/k$. Binomial SE at
$k=120$ is 0.046, inside every gap in the grid, and $k=120$ clears the largest
required $k$ (47) by a factor of 2.5.

**Secondary: verbalized probability**, one call per cell per condition. No
sampling noise, so it is the only channel that can see the paper's 0.02 effect,
but Section 4 of paper 2 treats verbalized confidence as an idealized function of
the collapsed predictive, and it suffers round-number attraction. Reported, never
primary.

## Conditions

**Known mixture (primary).** Both biases and the uniform prior are in the prompt,
so the Bayes answer is determined and a deviation is unambiguously a failure.

**Unknown mixture (secondary).** No structure given. The typed path is not
uniquely defined here, so this arm is descriptive and read against laplace.

## The two controls

**Arithmetic control.** In the known-mixture condition only, one call per cell
asking for the posterior weight on the high-bias coin. This separates the two
readings of a freeze: a model that reports the posterior correctly and still
forecasts 0.5 has collapsed, which is the paper's phenomenon; a model that cannot
compute the posterior is showing a capability failure the paper does not claim.
Without this the headline result is unattributable.

**Mode-collapse detector.** Sampled frequency estimates a forecast only if the
model samples. A model answering its argmax gives $\hat p \in \{0,1\}$, so the
channel measures the sign of a deviation and not its size. On the null cells all
three paths predict 0.5, so a sampling model lands near 0.5 and an
argmax-answering one does not. The detector fires if either null cell has
$\hat p < 0.20$ or $\hat p > 0.80$. This is not hypothetical: the `collapsed`
dry run is scored `laplace` with a decisive likelihood ratio of 62 by the
scorer alone, and only the two guards below overturn it.

## Scorer

Log-likelihood of the observed head count under $\mathrm{Binomial}(k, p_\text{path})$,
summed over the eight discriminating cells. Null cells are excluded from the sum
because they cannot discriminate. Reported as a ranking with the log-likelihood
ratio of the winner over the runner-up.

Two guards against a confidently wrong verdict:

- **Fit.** $G^2 = 2(\ell_\text{saturated} - \ell_\text{winner})$ against the 1%
  point of $\chi^2_8$ (Wilson-Hilferty, cut 20.12). Above the cut, the verdict is
  `none`: no registered path describes the data.
- **Mode collapse.** As above. If it fires, the sampled channel is reported as
  invalid and the verbalized channel becomes primary.

The E5 slope of `coin_example.py` is computed on the verbalized channel and
reported as a secondary statistic for continuity with the paper. It is not the
decision statistic, for three reasons: three points with an intercept leave
residual df 1, the $d{=}0.30$ point carries leverage 0.94, and the dry run shows
it scores a **frequency-counting model as slope 0.000**, which reads as "frozen".
The statistic cannot see the third mode at all.

## Registered thresholds

| quantity | value |
|---|---|
| $k$, sampled channel | 120 |
| decisive log-likelihood ratio | 10 |
| $G^2$ cut for `none` | 1% point of $\chi^2_8$ = 20.12 |
| mode-collapse cut on null cells | $\hat p < 0.20$ or $\hat p > 0.80$ |
| power target for the grid | 80% at 5% two-sided |

## Predictions

1. **Freeze (what the paper predicts).** In the known-mixture saturation cells
   the flat path wins, with a log-likelihood ratio over the runner-up of at
   least 10.
2. **Typed.** The typed path wins those cells at the same margin. This is the
   outcome that counts against the paper.
3. **Counting.** The laplace path wins the known-mixture saturation cells. The
   model is neither frozen nor typed, and the paper's two-way framing is the
   wrong description of what deployed systems do. This prediction exists so that
   outcome is a registered finding rather than an improvisation.
4. **Unknown-mixture arm.** Laplace beats typed, since the prompt supplies no
   mixture to be typed about. A typed win here would mean the model is importing
   a two-coin structure that was never stated.

Predictions 1 through 3 are exhaustive over the registered paths, with `none` as
the fourth outcome. Fixing all four in advance is what stops any result from
being read as confirmation after the fact.

## What would count against the paper

Prediction 2 in the known-mixture condition, with the arithmetic control showing
the model also computes the posterior correctly. That says behavioral and
verbalized reporting are not collapsing the model channel on this task, and the
freeze does not reproduce on a deployed system at this scale.

Prediction 3 is a weaker but still adverse result: the phenomenon is real in the
sense that the model is not typed, but the paper names the wrong failure mode.

Either is worth reporting, and both stay in the paper.

## Budget

10 sampled cells × 120 × 2 conditions = 2,400 single-token completions, plus 39
verbalized and control calls. Total 2,439. $k$ may be reduced to 60 and still
clear the largest required $k$ of 47; that halves the run and is the only change
to this registration that does not require re-registering.

## Recorded at run time

Model id and version, timestamp per call, the exact prompt length, and every raw
response string, in one JSONL under `results/`. Re-analysis reads the JSONL, so
no claim in the writeup depends on a run being repeated. The paper's empirical
claims describe the runs performed and do not extrapolate past them.
