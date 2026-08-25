# E-C: what the exploratory run established, and what a confirmatory run needs

## Bottom line

E-C supports the hypothesis in both models tested, and is not confirmatory.

The hypothesis: uncertainty about which reading of a question applies predicts the value
of clarification better than either a scalar confidence score or undifferentiated
variability in the model's answers.

All four directional predictions held in Qwen3-8B and Llama-3-70B, with the main
intervals excluding zero. What keeps it exploratory is the protocol, not the result: it
was written with the models and the matched sets already in hand, one stochastic run per
model, mechanical alias clustering in place of a semantic audit.

## Evidence

### Ambiguous questions carry measurable between-reading variance

| Model | Set A minus Set B between-reading variance | 95% CI |
|---|---:|---:|
| Qwen3-8B | 0.0858 | [0.0378, 0.1438] |
| Llama-3-70B | 0.1812 | [0.1213, 0.2414] |

Set B has one reading, so its between-reading variance is zero by construction.

### Clarification helps the ambiguous set more than the control

| Model | Set A gain | Set B gain | A minus B | 95% CI |
|---|---:|---:|---:|---:|
| Qwen3-8B | 0.1005 | 0.0240 | 0.0766 | [0.0083, 0.1526] |
| Llama-3-70B | 0.1218 | 0.0028 | 0.1190 | [0.0497, 0.1950] |

Qwen's Set B figure is a repeat-batch control on identical prompts, and its
0.0240 is discussed under the caveats below.

### Between-reading variance predicts which items benefit

The reported estimate is split-half, because the same 32 samples estimated both the
predictor and the outcome and shared sampling noise inflates a same-sample correlation.
The variance comes from 16 draws and the gain from the disjoint 16, with both
half-assignments reported since neither half is privileged.

| Model | Split | Pearson r | 95% CI |
|---|---|---:|---:|
| Qwen3-8B | half 1 | 0.684 | [0.103, 0.934] |
| Qwen3-8B | half 2 | 0.829 | [0.499, 0.946] |
| Llama-3-70B | half 1 | 0.582 | [0.334, 0.787] |
| Llama-3-70B | half 2 | 0.610 | [0.370, 0.807] |

All four exclude zero. The same-sample figures, 0.790 [0.340, 0.951] for Qwen and 0.613
[0.375, 0.815] for Llama, are inflated by the shared draws and belong in an appendix.

### The decomposition is what carries the signal

| Model | Predictor of gain | Pearson r | 95% CI |
|---|---|---:|---:|
| Qwen3-8B | between-reading variance (same sample) | 0.790 | [0.340, 0.951] |
| Qwen3-8B | undifferentiated answer variance | 0.223 | [-0.386, 0.698] |
| Qwen3-8B | scalar confidence (within band) | -0.147 | [-0.355, 0.147] |
| Llama-3-70B | between-reading variance (same sample) | 0.613 | [0.375, 0.815] |
| Llama-3-70B | undifferentiated answer variance | -0.032 | [-0.336, 0.290] |
| Llama-3-70B | scalar confidence (within band) | -0.117 | [-0.368, 0.191] |

Paired bootstrap advantage over undifferentiated variance: 0.567 [0.186, 1.068] for Qwen
and 0.646 [0.312, 0.942] for Llama. A difference of two correlations can exceed one even
though each is bounded by one.

The comparison that carries weight is against undifferentiated answer variance, whose
predictor retains its full range. The confidence rows are within-band observations and
not a fair comparison: the design screened items into a 50 to 60 percent band, every
retained Qwen item rates 50 or 60, and a near-zero correlation inside a range that narrow
is the expected reading rather than evidence. The paired advantage over confidence is
reported nowhere for the same reason. The operative evidence against the score is E-B's
between-set contrast, matched confidence with unmatched value.

## Caveats on the design

- The E-C gain is oracle-style: for each item, the improvement from replacing the
  ambiguous prompt with a reading's annotated rewrite, averaged uniformly over the two
  readings, against a fresh 32-sample baseline. It requires the dataset's rewrites, so it
  is a benchmark quantity and not computable at inference time. It is a different
  estimand from the E-B gains, so the two sets of numbers are not to be differenced.
- The two readings are not exchangeable: the later-run reading batch scores 16.5 points
  below the earlier one for Qwen and 11.5 for Llama. The uniform average over readings is
  reported as defined, with the non-exchangeability as a caveat on it.
- Qwen's repeat-batch control came in at +2.4 points [+0.6, +4.8] on identical prompts.
  The drift explanation was checked and came back negative, so it is read as a probable
  chance fluctuation. A reversed-order control run is open.
- Clustering is mechanical alias matching, which can miss valid paraphrases.

## What a confirmatory run needs

1. Freeze and preregister the protocol and analysis unchanged before running.
2. Fresh held-out ambiguous and matched control questions.
3. Checkpoints and exclusion rules chosen before results are observed.
4. Additional model families, and a frontier model.
5. Independent seeds.
6. An annotation-blind clustering audit beside the mechanical aliases.

Item 6 has now run and is answered. The frozen clustering run of 2026-08-25
(`ec/experiment_results_1/EC_CLUSTERING_FULL_RESULTS.md`, repo commit `51ff250`) found
that annotation-blind labels do not preserve the correlation: every Q and QS interval
includes zero, and the agreement audit locates the failure in reading discovery rather
than in assignment or in the estimator. That result stands on its item slice, which was
the confidence-screened subset, and Experiment D re-asks it at full-pool scale.

What replaces item 6 on the checklist is the answer-side arm: cluster the saved samples
under bidirectional entailment, then report the entropy over cluster masses as the
semantic-entropy baseline beside the between-cluster variance as the estimator under
inferred labels. It is item 4 of the queue in `paper2_plan.md`.

Preserving the gain contrasts, the Set A correlations, and the paired advantage over
undifferentiated variance under that protocol would make the hypothesis confirmed.

## Sources

Every value above derives from the committed summaries in the repo
`github.com/george-adams1/latent_uncertainty_estimation_experiments`:
`scan_results/ec_qwen3_8b_results.jsonl.summary.json`,
`scan_results/ec_llama3_70b_results.jsonl.summary.json`, and the split-half rows from
`ec/split_half.py` run over the committed records. Full results in
`EC_EXPERIMENT_RESULTS.md`, operational protocol in `ec_protocol.md`.
