# E-A results: Llama-3-70B freeze test

## Executive summary

Experiment E-A was run on 25 August 2026 against
`meta-llama/Meta-Llama-3-70B-Instruct` using the preregistered sample size
`k = 120`. All 2,439 expected responses were recorded and all were parseable.

The preregistered verdict was **`none` in both experimental conditions**. In
each condition, the Laplace/counting path had the highest likelihood among the
three registered paths, but it failed the absolute goodness-of-fit test by a
large margin. The known-mixture sampled channel also triggered the registered
mode-collapse detector. These safeguards mean that the result must not be
reported as evidence for the Laplace path.

The secondary verbalized channel was more suggestive. When the two possible
coin biases were stated explicitly, the model reproduced the typed forecast in
the principal saturation cells and had an E5 slope of `1.017`, close to the
typed prediction of `1` and far from the frozen prediction of `0`. However, it
gave severely incorrect answers in other verbalized and arithmetic-control
cells. The run therefore does not establish that the model is a consistently
typed reasoner.

The defensible conclusion is that this Llama-3-70B run **did not reproduce the
predicted freeze, but also did not cleanly follow the typed or counting path**.
Its sampled behavior did not function as a calibrated probabilistic forecast,
and the preregistered design correctly returned `none` rather than assigning a
misleading winner.

## Run configuration

| Field | Value |
|---|---|
| Experiment | E-A freeze test |
| Model | `meta-llama/Meta-Llama-3-70B-Instruct` |
| Cached model revision | `50fd307e57011801c7833c87efa1984ddf2db42f` |
| Date | 2026-08-25 UTC |
| Slurm allocation | `93675` on `dgx-02` |
| Hardware | 8 x NVIDIA H100 80GB |
| Inference server | vLLM 0.24.0, tensor parallel size 8 |
| Sample size | `k = 120` per sampled cell |
| Conditions | Known mixture and unknown mixture |
| Grid | 13 cells: 10 sampled, all 13 verbalized |
| Sampled responses | 2,400 |
| Verbalized responses | 26 |
| Arithmetic-control responses | 13 |
| Total responses | 2,439 |
| Unparseable responses | 0 |

The prompts, grid, sample size, thresholds, and scorer were unchanged from
`freeze_test_prereg.md`. The only new harness code was transport support for a
local OpenAI-compatible vLLM endpoint.

## Registered decision rule

The sampled channel compares three prespecified paths:

- **Typed:** update a posterior over which of the two stated coins was selected,
  then marginalize the next-flip probability.
- **Flat:** remain at probability `0.5`, representing the predicted freeze.
- **Laplace:** use the frequency-counting estimate `(s + 1) / (n + 2)` without
  respecting the stated bounds on the coin biases.

The path with the largest binomial log-likelihood is only accepted if it also
passes the registered absolute goodness-of-fit test. The verdict becomes `none`
when `G²` exceeds the 1% cutoff of `20.12`. Independently, sampled rates below
`0.20` or above `0.80` in either null cell trigger the mode-collapse detector,
because all three paths predict `0.5` there.

## Primary sampled-channel results

| Condition | Laplace LL | Typed LL | Flat LL | Nominal winner | Winner-vs-runner-up LLR | G² | Mode-collapse guard | Registered verdict |
|---|---:|---:|---:|---|---:|---:|---|---|
| Known mixture | -226.67 | -289.11 | -665.42 | Laplace | 62.44 | 453.34 | Triggered | **`none`** |
| Unknown mixture | -254.40 | -350.99 | -665.42 | Laplace | 96.59 | 293.00 | Not triggered | **`none`** |

Both likelihood ratios exceed the registered decisive threshold of `10`, but
this does not override the fit test. The `G²` values are roughly 23 and 15
times the cutoff, respectively. No registered path adequately describes the
full pattern.

### Why the known-mixture sampled channel was invalid

The known-mixture samples were nearly deterministic in the decisive cells:

| Cell type | Observed head rates |
|---|---|
| Four saturation cells | `0.000`, `1.000`, `0.000`, `1.000` |
| Two extreme cells | `0.000`, `1.000` |
| Two paper cells | `0.000`, `1.000` |
| Two null cells | `0.808`, `0.900` |

All registered paths predict `0.5` in the null cells. Rates of `0.808` and
`0.900` cross the registered mode-collapse threshold and show that repeated
`H`/`T` generations were not behaving as samples from the model's stated
forecast. The raw likelihood scorer preferred Laplace because the other cells
also tended toward `0` or `1`, but the null cells and absolute fit statistic
correctly prevented that superficial resemblance from becoming the verdict.

The unknown-mixture null cells were well behaved (`0.533` and `0.433`), so its
mode-collapse guard did not fire. Nevertheless, the full pattern still failed
the goodness-of-fit test decisively.

## Secondary verbalized channel

### Known-mixture saturation cells

The cleanest positive result occurred when the mixture was explicit and the
history consisted entirely of heads or entirely of tails:

| Cell | Typed | Stated |
|---|---:|---:|
| `d=0.20, n=10, s=0` | 0.300 | 0.300 |
| `d=0.20, n=10, s=10` | 0.700 | 0.700 |
| `d=0.20, n=20, s=0` | 0.300 | 0.300 |
| `d=0.20, n=20, s=20` | 0.700 | 0.700 |

These are exactly the bounded, saturating forecasts that distinguish typed
reasoning from unrestricted frequency counting.

### Paper sweep

For the paper's one-head sweep, the known-mixture stated probabilities were
`0.500`, `0.520`, and `0.680` at `d = 0.05`, `0.10`, and `0.30`. The fitted E5
slope was:

| Condition | E5 slope | Typed reference | Frozen reference |
|---|---:|---:|---:|
| Known mixture | 1.017 | 1 | 0 |
| Unknown mixture | -1.284 | 1 | 0 |

The known-mixture slope is close to typed, but it is a three-point secondary
statistic with no error bar and does not override the primary `none` verdict.
The unknown-mixture values (`0.500`, `1.000`, `0.500`) were erratic and produced
an uninterpretable negative slope.

### Inconsistencies and arithmetic controls

The model did not apply the same calculation reliably across the grid. Selected
failures include:

| Cell | Quantity requested | Exact value | Model response |
|---|---|---:|---:|
| Known extreme, `d=0.40, n=10, s=3` | Next-head probability | approximately 0.100 | 0.900 |
| Known null, `d=0.20, n=10, s=5` | Next-head probability | 0.500 | 0.700 |
| Known null, `d=0.40, n=10, s=5` | Next-head probability | 0.500 | 0.900 |
| Extreme arithmetic control, `d=0.40, n=10, s=3` | Posterior weight on high-bias coin | approximately 0.00015 | 0.9242 |
| Null arithmetic control, `d=0.20, n=10, s=5` | Posterior weight on high-bias coin | 0.500 | 0.831472 |
| Null arithmetic control, `d=0.40, n=10, s=5` | Posterior weight on high-bias coin | 0.500 | 0.9477 |

Other simple controls were accurate or close: the one-flip controls at
`d=0.30` were `0.2` and `0.8095` against exact values `0.2` and `0.8`, and the
small-spread one-head controls at `d=0.05` and `d=0.10` were `0.551` and `0.6`
against `0.55` and `0.6`.

The mixture of exact successes and large reversals means the arithmetic control
does not validate consistent posterior computation. Consequently, the typed
verbalized results cannot be treated as the clean typed outcome described in
the preregistration.

## Interpretation relative to the preregistered predictions

1. **Freeze was not observed.** The flat path was the worst of the three
   likelihood fits in both conditions, and the known-mixture verbalized slope
   was close to the typed value rather than zero.
2. **Typed reasoning was not established.** The primary channel returned
   `none`, and important verbalized and arithmetic-control cells were wrong.
3. **Counting was not established.** Laplace was the nominal likelihood winner,
   but failed the registered absolute fit test severely. Reporting this as a
   Laplace verdict would violate the preregistration.
4. **The main empirical finding is an elicitation mismatch.** Asking the model
   to simulate a flip and sampling its output tokens did not consistently
   recover its verbalized probability, especially when the coin mixture was
   explicit. The null-cell guard was included for exactly this failure mode.

This is therefore a valid `none` result rather than a failed run: the harness
completed, its controls detected that the sampled channel was not measuring the
intended quantity, and its decision rule declined to force the observations
into an inadequate theoretical category.

## What can be claimed

A concise report suitable for the paper is:

> On Llama-3-70B, neither the typed, frozen, nor frequency-counting path fit the
> preregistered E-A sampled forecasts. Although frequency counting was the
> nominal likelihood winner, it failed the absolute fit test in both conditions,
> and the known-mixture null cells triggered the mode-collapse guard. The
> registered verdict was therefore `none` in both conditions. Verbalized
> forecasts matched the typed path in the principal saturation cells and on the
> paper's slope sweep, but substantial errors on other cells and arithmetic
> controls prevented a clean typed interpretation.

The run should not be described as demonstrating that Llama-3-70B is frozen,
typed, or Laplace-counting. It shows that the deployed model can verbalize the
typed calculation in some salient cases while its repeated categorical outputs
and broader arithmetic behavior remain inconsistent with all three registered
idealizations.

## Provenance

- Preregistration and decision thresholds: `freeze_test_prereg.md`
- Harness: `freeze_test.py`
- Raw records:
  `results/freeze_meta-llama/Meta-Llama-3-70B-Instruct_20260825T020545.jsonl`
- Complete scored output: `results/ea_llama3_70b_93675.run.log`
- Server/model-loading record: `results/llama3_70b_vllm_server_93675.log`
