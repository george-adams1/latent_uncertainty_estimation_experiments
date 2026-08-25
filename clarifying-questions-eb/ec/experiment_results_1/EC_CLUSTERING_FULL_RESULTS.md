# E-C inference-time clustering: frozen full results

Run date: 2026-08-25 UTC

Status: **frozen confirmatory run complete**

## Question and design

This arm tests whether E-C's between-reading variance remains predictive when
answer clusters are inferred without AmbigQA annotations. It reuses the saved
32-sample E-C records; it does not resample either subject model.

The fixed clusterer was
`meta-llama/Meta-Llama-3-70B-Instruct` at immutable revision
`50fd307e57011801c7833c87efa1984ddf2db42f`, served by vLLM 0.24.0 at
temperature 0 and seed 0. It ran tensor-parallel over all eight H100s on
`dgx-26`, Slurm job `91038`, step `38`. The prompts were frozen in commit
`79554bc` before this run.

Two annotation-blind variants were evaluated:

- **Q:** infer interpretations from the question alone, then assign each saved
  answer to an inferred interpretation or `none`.
- **QS:** infer interpretations from the question plus the 32 ambiguous/original
  samples, then assign answers in the same way.

The oracle arm and normalized answer-string arm use the identical E-C estimator.
The final pilot's five Set A and five Set B IDs per subject model were excluded.
The confirmatory samples are therefore 25 Set A + 25 Set B items for Qwen3-8B
and 39 Set A + 40 Set B items for Llama-3-70B.

Every number below is derived from:

- `ec/experiment_results_1/ec_clustering_full_qwen3_8b.jsonl.summary.json`
- `ec/experiment_results_1/ec_clustering_full_llama3_70b.jsonl.summary.json`

The corresponding JSONLs in `ec/experiment_results_1/` contain all per-item
lister outputs, per-response assignments, estimator values, agreement audits,
and provenance. Their embedded `results_path` fields retain the original
run-time paths under `scan_results/`; the files were archived here afterward
without modifying their contents or hashes.

## Headline result

Inference-time clustering does **not** preserve the oracle predictor in this
run. All Q and QS 95% confidence intervals include zero. On Qwen responses, the
same-sample correlation falls from 0.804 under oracle labels to 0.236 with Q and
0.209 with QS. On Llama responses, it falls from 0.569 to 0.298 with Q and 0.130
with QS.

Pearson correlations below use Set A items. Intervals are deterministic
item-level percentile bootstrap intervals with 10,000 resamples, seed 0.

| Subject responses | Labels | Same 32 | First-half predictor | Second-half predictor |
|---|---|---:|---:|---:|
| Qwen3-8B | Oracle | 0.804 [0.269, 0.969] | 0.698 [-0.024, 0.951] | 0.850 [0.466, 0.984] |
| Qwen3-8B | Q | 0.236 [-0.206, 0.644] | 0.275 [-0.188, 0.658] | 0.172 [-0.242, 0.604] |
| Qwen3-8B | QS | 0.209 [-0.207, 0.625] | 0.223 [-0.209, 0.620] | 0.185 [-0.214, 0.584] |
| Qwen3-8B | Answer string | 0.230 [-0.216, 0.631] | 0.182 [-0.263, 0.584] | 0.254 [-0.200, 0.646] |
| Llama-3-70B | Oracle | 0.569 [0.271, 0.811] | 0.534 [0.224, 0.782] | 0.564 [0.282, 0.805] |
| Llama-3-70B | Q | 0.298 [-0.060, 0.623] | 0.283 [-0.067, 0.617] | 0.314 [-0.050, 0.623] |
| Llama-3-70B | QS | 0.130 [-0.275, 0.528] | 0.129 [-0.252, 0.481] | 0.130 [-0.283, 0.536] |
| Llama-3-70B | Answer string | 0.321 [0.009, 0.586] | 0.324 [0.003, 0.590] | 0.300 [-0.009, 0.567] |

The answer-string correlation on Llama should not be interpreted as a successful
reading estimator. Its Set B diagnostics below show that it clusters ordinary
answer variation as if it were latent structure.

## Q versus QS detection tradeoff

Counts below are items for which the lister returned more than one inferred
interpretation.

| Subject responses | Variant | Set A | Set B |
|---|---|---:|---:|
| Qwen3-8B | Q | 20/25 (80.0%) | 6/25 (24.0%) |
| Qwen3-8B | QS | 19/25 (76.0%) | 15/25 (60.0%) |
| Llama-3-70B | Q | 20/39 (51.3%) | 9/40 (22.5%) |
| Llama-3-70B | QS | 18/39 (46.2%) | 18/40 (45.0%) |

QS did not deliver the hoped-for recall/precision tradeoff. It detected no more
Set A ambiguity than Q, but approximately doubled or more than doubled the Set B
multi-interpretation rate. Q is the less damaging inference-time variant here,
although its predictive intervals still include zero.

## Agreement audit

Inferred IDs were aligned to oracle reading labels with the maximum-overlap
one-to-one mapping, used only after inference. Agreement on responses uniquely
matched to one oracle reading is high, but the pipeline recovers all oracle
readings on only one item per subject model. This distinguishes good assignment
conditional on a usable list from failure to discover the full reading set.

| Subject responses | Variant | All-response agreement | Oracle-unique agreement | Items recovering all readings | Set A `none` rate |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Q | 33.1% | 90.0% (307/341) | 1/25 (4.0%) | 21.7% |
| Qwen3-8B | QS | 29.9% | 93.5% (319/341) | 1/25 (4.0%) | 17.5% |
| Llama-3-70B | Q | 42.1% | 86.3% (1422/1648) | 1/39 (2.6%) | 5.9% |
| Llama-3-70B | QS | 39.0% | 83.7% (1379/1648) | 1/39 (2.6%) | 4.1% |

The oracle matcher itself labels 2,059/2,400 Qwen Set A responses and
1,914/3,744 Llama Set A responses as `other`; Llama also has 182 `multiple`
labels. All-response agreement is therefore not a pure reading-assignment
accuracy measure.

## Variance levels and Set B honesty control

Mean Set A between-reading variance:

| Subject responses | Oracle | Q | QS | Answer string |
|---|---:|---:|---:|---:|
| Qwen3-8B | 0.0841 | 0.1108 | 0.1390 | 0.1646 |
| Llama-3-70B | 0.1802 | 0.0797 | 0.0956 | 0.2494 |

For Set B, the paper's between-reading estimand remains structurally zero because
there is only one reading-conditioned distribution. Two separate diagnostics
measure false structure without redefining that estimand:

| Subject responses | Arm | Mean observed categorical variance | Mean between-batch variance [95% CI] |
|---|---|---:|---:|
| Qwen3-8B | Oracle | 0.0761 | 0.00221 [0.00008, 0.00486] |
| Qwen3-8B | Q | 0.1230 | 0.00100 [0.00029, 0.00211] |
| Qwen3-8B | QS | 0.1477 | 0.00162 [0.00070, 0.00270] |
| Qwen3-8B | Answer string | 0.5680 | 0.01137 [0.00693, 0.01674] |
| Llama-3-70B | Oracle | 0.0406 | 0.00027 [0.00001, 0.00071] |
| Llama-3-70B | Q | 0.0484 | 0.00088 [0.00004, 0.00227] |
| Llama-3-70B | QS | 0.0813 | 0.00081 [0.00018, 0.00162] |
| Llama-3-70B | Answer string | 0.2396 | 0.00500 [0.00242, 0.00796] |

`between_batch_variance` is finite-sample separation between the original and
repeat Set B batches, not between-reading variance. The answer-string arm is much
larger on both Set B diagnostics, confirming that surface-form clustering
manufactures apparent structure from ordinary answer scatter.

## Interpretation

This run supports the protocol's stated kill condition: removing annotation-based
reading labels drags the E-C correlations toward the answer-string baseline. The
oracle result remains positive, especially in the same-sample analyses, so the
experiment does not refute the estimator conditional on correct reading clusters.
It identifies inference-time reading discovery as the current competence
bottleneck.

The most defensible paper statement is therefore: E-C predicts clarification value
when evaluated with oracle reading labels, but this fixed Llama-70B inference-time
clusterer does not retain a statistically distinguishable correlation on the held-
out non-pilot items. Q is preferable to QS in this implementation because QS adds
Set B false positives without increasing Set A multi-reading detection.

This is one recorded deterministic-intent realization. The pilot showed that
temperature-0 TP8 vLLM calls with concurrency were not bit-identical across repeat
runs, so the saved JSONLs—not an assumption of bitwise determinism—define the run
of record.

## Artifact integrity

- Qwen JSONL SHA-256:
  `2d76915d8767a25e5180c19eb584f239ba6f480195bbff2ca5d1e101e720f229`
- Qwen summary SHA-256:
  `4e5c184f559818a6bc4de8a6246f634b6e949294cc2942f8e460750e06c23da3`
- Llama JSONL SHA-256:
  `1e61e974e6db055d0cdbe581a79eef1871a48bca371934ed19acc2393d639619`
- Llama summary SHA-256:
  `d0a0ffafa9f630342356501f3c30878b5733b20b7eb6d82f25bfe0f40ae84569`

Record-level validation checked counts, unique IDs, pilot disjointness, schema v2,
frozen-run marking, subject and clusterer models, immutable revision, prompt
commit, node/job/step provenance, expected batches, 32 assignments per batch,
and presence of the separate Set B diagnostic. All checks passed.
