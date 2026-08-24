# E-C variance-decomposition experiment results

Run date: 2026-08-20 UTC  
Node: `dgx-26`  
Hardware: 8 × NVIDIA H100 80GB HBM3  
Inference server: vLLM 0.24.0

Status: **completed exploratory experiment; not preregistered**. The estimand
and analysis were frozen in `ec_protocol.md` before these full E-C runs.

## Configuration

- Models: `Qwen/Qwen3-8B` and
  `meta-llama/Meta-Llama-3-70B-Instruct`.
- Qwen used eight data-parallel replicas, one per GPU. Llama used tensor
  parallelism across all eight GPUs.
- `VLLM_USE_FLASHINFER_SAMPLER=0` selected vLLM's standard stochastic sampler
  because this node session did not expose `nvcc` for FlashInfer's JIT sampler.
- Each original and fixed-reading prompt received 32 samples at temperature
  1.0.
- Qwen sample: 30 Set A and 30 Set B items.
- Llama sample: 44 Set A and 45 Set B items. One leaked Set A E-B record was
  excluded before E-C.
- Confidence intervals are deterministic item-level percentile-bootstrap
  intervals with 10,000 resamples, seed 0, and 95% confidence level.
- Answer clusters and correctness use the dataset alias lists mechanically;
  no judge model is involved.

The categorical variance is `1 - sum_c p(c)^2`. The between-reading component
is the part of the fixed-reading mixture's total variance explained by which
reading was selected. Set B has only one reading, so its between-reading
variance is zero by construction.

## Primary results

| Model | Mean between variance, A | Mean between variance, B | A−B (95% CI) | Mean gain, A | Mean gain, B | Gain A−B (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 0.0858 | 0 | 0.0858 [0.0378, 0.1438] | 0.1005 | 0.0240 | 0.0766 [0.0083, 0.1526] |
| Llama-3-70B | 0.1812 | 0 | 0.1812 [0.1213, 0.2414] | 0.1218 | 0.0028 | 0.1190 [0.0497, 0.1950] |

Here, gain is clarified accuracy minus baseline accuracy, averaged uniformly
over the two possible intended readings for Set A. Thus the gain differences
are approximately 7.66 percentage points for Qwen and 11.90 percentage points
for Llama.

Additional means:

| Model | Set | Observed ambiguous variance | Within-reading variance | Between fraction |
|---|---|---:|---:|---:|
| Qwen3-8B | A | 0.1006 | 0.0997 | 0.1906 |
| Qwen3-8B | B | 0.0775 | 0.0926 | 0 |
| Llama-3-70B | A | 0.0729 | 0.0963 | 0.3692 |
| Llama-3-70B | B | 0.0361 | 0.0326 | 0 |

## Item-level prediction within Set A

| Model | Predictor versus realized gain | Pearson r (95% CI) |
|---|---|---:|
| Qwen3-8B | Between-reading variance | 0.790 [0.340, 0.951] |
| Qwen3-8B | Undifferentiated ambiguous variance | 0.223 [−0.386, 0.698] |
| Qwen3-8B | Scalar confidence | −0.147 [−0.355, 0.147] |
| Llama-3-70B | Between-reading variance | 0.613 [0.375, 0.815] |
| Llama-3-70B | Undifferentiated ambiguous variance | −0.032 [−0.336, 0.290] |
| Llama-3-70B | Scalar confidence | −0.117 [−0.368, 0.191] |

The paired correlation comparisons were also positive in both models:

| Model | Correlation difference | Estimate (95% CI) |
|---|---|---:|
| Qwen3-8B | Between minus ambiguous variance | 0.567 [0.186, 1.068] |
| Qwen3-8B | Between variance minus confidence | 0.937 [0.353, 1.251] |
| Llama-3-70B | Between minus ambiguous variance | 0.646 [0.312, 0.942] |
| Llama-3-70B | Between variance minus confidence | 0.730 [0.374, 1.026] |

Bootstrap differences between correlations can exceed one even though each
individual correlation is bounded to `[-1, 1]`.

## Interpretation

Both models support the E-C predictions in this sample:

1. Set A has a positive between-reading variance component.
2. Clarification gain is higher for Set A than for the Set B repeat control.
3. Between-reading variance positively predicts realized clarification gain
   within Set A.
4. It predicts gain more strongly than either undifferentiated answer
   variance or scalar confidence.

These are exploratory estimates from one stochastic run. The Set B
between-reading value is structurally zero, so the Set A-only correlations are
the primary item-level prediction result. The alias-based clusters can miss
valid paraphrases that are absent from the dataset aliases.

## Artifacts

Qwen:

- `scan_results/ec_qwen3_8b_results.jsonl`
- `scan_results/ec_qwen3_8b_results.jsonl.summary.json`
- `scan_results/ec_qwen3_8b_run.log`
- `scan_results/ec_qwen3_8b_server.log`

Llama:

- `scan_results/ec_llama3_70b_results.jsonl`
- `scan_results/ec_llama3_70b_results.jsonl.summary.json`
- `scan_results/ec_llama3_70b_run.log`
- `scan_results/ec_llama3_70b_server.log`

All 60 Qwen records and all 89 leak-filtered Llama records use schema `ec.v1`.
The maximum recorded total-variance decomposition identity error is exactly
0.0 for both runs.
