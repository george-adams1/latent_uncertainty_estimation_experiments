# E-C operational protocol: variance estimator

Status: **implemented exploratory protocol; not preregistered**.

The top-level project plan names E-C as the “variance estimator” experiment
but does not include a complete E-C specification. This document records the
operational definition implemented in `ec/`. It must be reviewed and frozen
before any future run is described as confirmatory.

## Question

Can repeated model samples distinguish uncertainty caused by competing
interpretations from uncertainty that remains after an interpretation is
fixed, and does the estimated between-reading component predict the realized
benefit of clarification?

## Source sample

E-C consumes a completed E-B matched sample rather than screening again. It
uses the E-B result JSONL to select Set A and Set B IDs and the E-B Set B
screening JSONL to recover TriviaQA answer aliases. Leaked E-B Set A records
are excluded by default.

This means E-C inherits E-B's:

- 50–60% post-answer confidence band;
- exactly-two-reading AmbigQA restriction;
- selected model and dataset versions;
- mechanical answer-alias grading limitations.

## Sampling

The default is 32 independent samples per prompt at temperature 1.0.

For each Set A item, sample:

1. the original ambiguous question 32 times;
2. the rewrite for reading A 32 times;
3. the rewrite for reading B 32 times.

For each Set B item, sample:

1. the original question 32 times;
2. the identical question in a second independent batch of 32 samples.

The Set B repeat is a null control for finite-sample changes in accuracy. Set B
has one reading, so its population between-reading variance is zero by
construction.

All raw samples, questions, system/user prompts, aliases, temperatures, and
cluster assignments are retained in the result record.

## Mechanical answer clusters

Each response is normalized and checked against every reading's answer-alias
list. It is assigned to:

- the unique matching reading;
- `multiple` if it matches more than one reading;
- `other` if it matches no reading.

No semantic judge is used. This is reproducible but undercounts paraphrases
that are absent from the dataset's alias lists.

## Variance estimand

Let the categorical answer cluster be represented by a one-hot random vector
`X`, and let `W` be the intended reading with a uniform prior over the two
AmbigQA readings. For a categorical distribution `p`, define total variance as

```text
V(p) = 1 - sum_c p(c)^2.
```

This is the trace of the one-hot covariance matrix and the probability that
two independent draws have different clusters.

Reading-conditioned empirical distributions are estimated from the rewritten
prompts. Their uniform mixture gives:

```text
total    = V(sum_w P(w) p(. | w))
within   = sum_w P(w) V(p(. | w))
between  = total - within
```

The implementation also computes the between term directly as
`sum_w P(w) ||p(.|w) - p_mixture||^2` and records the numerical identity
error. It records the observed variance of the ambiguous-prompt samples and
the total-variation distance between that distribution and the constructed
reading mixture as model checks.

## Realized clarification gain

For each possible intended Set A reading:

- baseline accuracy is the fraction of ambiguous-prompt samples strictly
  correct for that reading;
- clarified accuracy is the fraction of that reading's rewritten-prompt
  samples strictly correct;
- reading-level gain is clarified accuracy minus baseline accuracy.

The item-level realized gain is the uniform mean over both readings. This
avoids depending on one randomly chosen intended reading.

For Set B, realized gain is repeat-batch accuracy minus original-batch
accuracy. Its expectation is zero; deviations quantify sampling noise.

## Predictions

1. Mean between-reading variance is higher on Set A than Set B.
2. Mean realized gain is higher on Set A than Set B.
3. Between-reading variance predicts realized gain, especially within Set A.
4. Between-reading variance predicts gain better than undifferentiated
   ambiguous-prompt variance or scalar confidence.

The analyzer reports point estimates and deterministic item-level percentile
bootstrap confidence intervals. It reports the prediction-correlation
differences needed for prediction 4, rather than asking the reader to compare
two correlations informally. Set contrasts and pooled correlations are
bootstrapped separately within Set A and Set B.

## CLI

Against an OpenAI-compatible inference server:

```bash
python -m ec.run_experiment \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --eb-results scan_results/full_eb_qwen3_8b_results.jsonl \
  --setb-screen scan_results/full_eb_qwen3_8b_setb_screen.jsonl \
  --samples-per-prompt 32 \
  --temperature 1.0 \
  --workers 32 \
  --retries 3 \
  --out ec_results_qwen3_8b.jsonl
```

Reanalyze saved records without inference:

```bash
python -m ec.analyze ec_results_qwen3_8b.jsonl
```

Use `--limit-per-set 1` for an inference smoke test. Use `--mock typed` with
fixture-shaped inputs for client-free development. A local Transformers model
is supported when `--base-url` and `--mock` are omitted, but then `--workers`
must remain 1.

## Interpretation limits

- This is an operationalization of the named E-C idea, not a recovered or
  previously frozen protocol.
- The Set B between-reading value is structurally zero, so pooled correlations
  can be driven by set membership. Set A-only correlations are therefore
  primary for item-level prediction.
- Samples from one question are not independent experimental items. Inference
  resamples questions, not individual completions.
- The estimator measures variability in alias-defined answer clusters, not all
  semantic variation in free-form generations.
