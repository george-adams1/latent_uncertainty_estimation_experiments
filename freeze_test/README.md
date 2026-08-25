# Code for "Collapsing Uncertainty: An Operator View of Second-Order Belief"

Adam M. Oberman (McGill University; Mila, Quebec AI Institute; LawZero).

Three self-checking scripts. Each exits 0 if and only if every assertion passes,
so running them is the verification.

```bash
python coin_example.py
python voi_check.py
python freeze_test.py --selfcheck
```

Requirements: Python 3 with NumPy. `coin_example.py` uses matplotlib headlessly if
it is importable, to write `coin_horizon.png`; its absence does not break the run.
The random seed is fixed (`np.random.default_rng(20260702)`), so runs reproduce.

## What `coin_example.py` checks

| | claim |
|---|---|
| E1 | the one-head closed form $m + v/m = 0.52$ |
| E2 | the trajectory table: typed drifts to the drawn bias, collapsed stays at 0.50 |
| E3 | the spread sweep $0.5 + 2d^2$, giving 0.505, 0.520, 0.680 |
| E4 | the horizon contrast, bimodal typed predictive against the tight flat bump |
| E4b | the Limit box coverage: 20,000 draws at $N = 2000$, coverage 0.0000 |
| E5 | the offline slope harness, mock reasoners scoring slope 0 and slope 1 |
| E6 | the coupled coins: 0.52 and 0.48 after one American head, $\mathrm{Cov} = -0.01$ |
| E7 | the two readings of the collapse operator on the pair (Section 2.6) |
| E8 | whether split conformal repairs the block-coverage failure (Section 4.2) |

E5 leaves `llm_predict(history, biases)` as the single hook for a real model. It
raises `NotImplementedError` by default and no API dependency is imported; the
two mock reasoners let the harness be validated offline against known answers.
Both arguments matter: the slope sweep varies the biases while holding the
history fixed, so a hook that ignored them would score every model as frozen.

## What `voi_check.py` checks

The value-of-information closed forms by exact enumeration: one flip is worth
$0.1$ under the identification utility and $0.0004$ under a one-step Brier
forecasting utility.

## What `freeze_test.py` does

Experiment E-A, the freeze test on a deployed model. Pre-registered in
`freeze_test_prereg.md`. It scores a model's forecasts against three exact paths,
not two: typed (bounded, saturating), flat (the collapse prediction), and Laplace
(frequency counting, unbounded). The third is why the grid is not the paper's:
the paper's one-head cells are either below the noise floor of the sampled
channel or sit on top of the counting path.

```bash
python freeze_test.py --selfcheck                     # offline, verifies the design
python freeze_test.py --dry-run --mock typed          # also: flat, laplace, collapsed
python freeze_test.py --run --model MODEL --k 120     # needs a key; wires call_model
python freeze_test.py --analyze results/FILE.jsonl    # re-score, costs nothing
```

For a local OpenAI-compatible server such as vLLM, set `OPENAI_BASE_URL` to
the server's `/v1` endpoint before using `--run`. This route does not require an
API key.

The completed Llama-3-70B run and its interpretation are documented in
[`EA_LLAMA3_70B_RESULTS.md`](EA_LLAMA3_70B_RESULTS.md).

The four dry runs are the check that matters before spending tokens: each drives
the full pipeline against a seeded mock emitting real-format responses, and the
scorer must recover the generating path. The `collapsed` mock is a model that
answers its argmax instead of sampling; the scorer alone calls it "laplace"
decisively, and the run passes only because the fit test and the null-cell
detector overturn that to "none".
