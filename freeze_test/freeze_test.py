"""E-A, the freeze test: does a deployed model update like a typed reasoner?

Companion to `coin_example.py`, which holds the theory and its offline checks.
This file is the real-model harness, kept separate because `coin_example.py` may
not carry an API dependency (see `coin_example_spec.md`). Nothing here imports an
SDK at module level either: the import happens inside `call_model`, so every
mode below runs with numpy alone.

WHAT IS BEING MEASURED

A coin is drawn once, uniformly, from two coins with biases (0.5-d, 0.5+d), then
flipped repeatedly. The model sees `s` heads in `n` flips and forecasts the next
one. Three reference paths are exact and known in advance:

    typed    posterior over *which coin*, then marginalize.  Bounded in
             (0.5-d, 0.5+d), so it SATURATES at the extreme bias.
    flat     0.5 forever. This is the collapse operator C: the paper's
             prediction for a reasoner that reports one confidence number.
    laplace  (s+1)/(n+2). Frequency counting: the "moving but wrong" third
             mode. UNBOUNDED, so it runs past the typed path's ceiling.

The paper frames the outcome as typed-or-frozen. A deployed model has an obvious
third option, counting, and a design blind to it will call counting "typed". So
all three are scored, and a fourth verdict ("none") is available.

WHY NOT THE PAPER'S OWN CELLS

The paper's headline numbers are one-head updates at d = 0.05, 0.10, 0.30, giving
typed forecasts 0.505, 0.520, 0.680 against a flat 0.5. In the sampled channel
those are unmeasurable or confounded (`--selfcheck` prints the required sample
sizes):

    d=0.05, n=1, s=1   typed vs flat needs k ~ 78,000
    d=0.10, n=1, s=1   typed vs flat needs k ~  4,900
    d=0.30, n=1, s=1   typed 0.680 vs laplace 0.667 -- needs k ~ 9,800

The third is the trap: it separates typed from flat cheaply, so it looks like the
good cell, but it sits on top of the counting path. A counting model scores
"typed" there. The saturation cells (long runs at moderate d) separate all three
at once, because the typed path is bounded and the counting path is not:

    d=0.20, n=20, s=20   typed 0.700   flat 0.500   laplace 0.954

The paper's cells are still run, in the verbalized channel only, where there is
no sampling noise. That is the one channel that can see a 0.02 effect.

MODES

    python3 freeze_test.py --selfcheck
        Offline. Checks the paths against coin_example.TypedReasoner and the
        closed forms, prints the grid with required sample sizes. Exits 0 iff
        every assert passes.

    python3 freeze_test.py --dry-run --mock {typed,flat,laplace,collapsed}
        Drives the whole pipeline (prompt -> response -> parse -> cache ->
        score) against a seeded mock that emits real-format "H"/"T" strings.
        The scorer must recover the generating path. This is what proves the
        measurement before any token is spent.

    python3 freeze_test.py --run --model MODEL [--k 120]
        The real run. Needs `call_model` wired and a key in the environment.
        Appends to a JSONL under results/ and resumes by skipping cached calls.

    python3 freeze_test.py --analyze results/FILE.jsonl
        Re-score a completed run. Costs nothing.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np

from coin_example import TypedReasoner

# ---------------------------------------------------------------------------
# Fixed constants. These are pre-registered in freeze_test_prereg.md; changing
# one after a run has been scored invalidates the registration.
# ---------------------------------------------------------------------------
Z_ALPHA_2 = 1.959963985      # two-sided 5%
Z_BETA = 0.841621234         # 80% power
Z_99 = 2.326347874           # one-sided 1%, for the none-of-the-above cut
K_DEFAULT = 120              # samples per cell in the sampled channel
MODE_COLLAPSE_CUT = 0.20     # null-cell p_hat outside [cut, 1-cut] => collapsed
LLR_DECISIVE = 10.0          # log-likelihood ratio the registration calls decisive
HISTORY_SEED = 20260810      # fixes the arrangement of heads within a history

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# The three reference paths. All exact, all offline, all functions of (d, n, s).
# ---------------------------------------------------------------------------
def typed_path(d, n, s):
    """Mixture Bayes: reweight the two coins by likelihood, then marginalize.

    Bounded strictly inside (0.5-d, 0.5+d). That ceiling is the signature the
    experiment exploits: no amount of evidence pushes it past the extreme bias.
    """
    r = TypedReasoner(biases=[0.5 - d, 0.5 + d], weights=[0.5, 0.5])
    for _ in range(s):
        r.update(1)
    for _ in range(n - s):
        r.update(0)
    return r.predict()


def flat_path(d, n, s):
    """The collapse operator C: the prior predictive, frozen. Ignores its past."""
    return 0.5


def laplace_path(d, n, s):
    """Frequency counting with a Beta(1,1) prior. Unbounded; ignores the coins."""
    return (s + 1) / (n + 2)


PATHS = {"typed": typed_path, "flat": flat_path, "laplace": laplace_path}


def typed_posterior_high(d, n, s):
    """Posterior weight on the high-bias coin. The arithmetic control's answer."""
    th = np.array([0.5 - d, 0.5 + d])
    lw = np.log(0.5) + s * np.log(th) + (n - s) * np.log(1.0 - th)
    w = np.exp(lw - lw.max())
    return float((w / w.sum())[1])


# ---------------------------------------------------------------------------
# Power. How many samples a cell needs before it can tell two paths apart.
# ---------------------------------------------------------------------------
def required_k(p1, p2, z_alpha_2=Z_ALPHA_2, z_beta=Z_BETA):
    """Samples to distinguish an observed rate p1 from a known reference p2.

    One-sample normal approximation at 5% two-sided, 80% power. Returns inf when
    the two paths coincide, which is the honest answer: no k suffices.
    """
    if abs(p1 - p2) < 1e-12:
        return float("inf")
    num = z_alpha_2 * np.sqrt(p2 * (1 - p2)) + z_beta * np.sqrt(p1 * (1 - p1))
    return float(np.ceil((num / (p1 - p2)) ** 2))


# ---------------------------------------------------------------------------
# The grid. Cells are (family, d, n, s); families decide how each cell is read.
# ---------------------------------------------------------------------------
def make_cell(family, d, n, s):
    t, f, l = typed_path(d, n, s), flat_path(d, n, s), laplace_path(d, n, s)
    return {
        "id": f"{family}_d{d:.2f}_n{n}_s{s}",
        "family": family, "d": d, "n": n, "s": s,
        "typed": t, "flat": f, "laplace": l,
        "gap_tf": abs(t - f), "gap_tl": abs(t - l), "gap_fl": abs(f - l),
        "min_gap": min(abs(t - f), abs(t - l), abs(f - l)),
        "k_tf": required_k(t, f), "k_tl": required_k(t, l),
    }


def select_grid():
    """The pre-registered grid.

    saturation  the decisive three-way test. Long runs at moderate spread, where
                the typed ceiling and the unbounded counting path pull apart.
                s=0 and s=n are mirror images, so a systematic H-over-T token
                preference shows up as an asymmetry between the pair.
    extreme     the opposite geometry: typed near 0.1/0.9, counting moderate.
                Guards against a scorer that only works in one direction.
    paper       the paper's own one-head cell. Read typed-vs-flat only; it
                cannot separate typed from counting.
    null        all three paths agree at 0.5. The harness control, and the
                mode-collapse detector.
    slope       the paper's three-point sweep, verbalized channel only.
    """
    cells = []
    for n in (10, 20):
        for s in (0, n):
            cells.append(make_cell("saturation", 0.20, n, s))
    for s in (3, 7):
        cells.append(make_cell("extreme", 0.40, 10, s))
    for s in (0, 1):
        cells.append(make_cell("paper", 0.30, 1, s))
    cells.append(make_cell("null", 0.20, 10, 5))
    cells.append(make_cell("null", 0.40, 10, 5))
    for d in (0.05, 0.10, 0.30):
        cells.append(make_cell("slope", d, 1, 1))
    return cells


SAMPLED_FAMILIES = ("saturation", "extreme", "paper", "null")
CONDITIONS = ("known", "unknown")


def history_string(cell, heads="H", tails="T"):
    """The flip history, as the prompt shows it.

    The arrangement of heads within the run is fixed by a seed rather than
    blocked, so recency effects are not systematically confounded with s. For
    s=0 and s=n there is only one arrangement and the seed does nothing.
    """
    n, s = cell["n"], cell["s"]
    seq = np.array([1] * s + [0] * (n - s))
    np.random.default_rng(HISTORY_SEED + n * 100 + s).shuffle(seq)
    return " ".join(heads if y else tails for y in seq)


# ---------------------------------------------------------------------------
# Prompts. Exact text is pre-registered; edit prereg and the run together.
# ---------------------------------------------------------------------------
def build_prompt(cell, condition, elicitation, heads="H", tails="T"):
    d = cell["d"]
    lo, hi = 100 * (0.5 - d), 100 * (0.5 + d)
    hist = history_string(cell, heads, tails)

    if condition == "known":
        setup = (
            f"A coin was selected once, at random, from two coins: one that lands "
            f"heads {lo:.0f}% of the time, and one that lands heads {hi:.0f}% of the "
            f"time. Each coin was equally likely to be selected. That same coin is "
            f"then flipped repeatedly."
        )
    elif condition == "unknown":
        setup = (
            "A coin of unknown bias is being flipped repeatedly. It is the same "
            "coin on every flip."
        )
    else:
        raise ValueError(f"unknown condition: {condition}")

    obs = (
        f"The flips so far, in order, were: {hist}"
        if cell["n"] > 0 else "No flips have been observed yet."
    )

    if elicitation == "sampled":
        ask = (
            f"Simulate the next flip of this same coin. Respond with exactly one "
            f"character, {heads} or {tails}, and nothing else."
        )
    elif elicitation == "verbalized":
        ask = (
            "What is the probability that the next flip of this same coin lands "
            "heads? Respond with exactly one number between 0 and 1, and nothing "
            "else."
        )
    elif elicitation == "control":
        ask = (
            f"Given the flips so far, what is the posterior probability that the "
            f"selected coin is the one that lands heads {hi:.0f}% of the time? "
            f"Respond with exactly one number between 0 and 1, and nothing else."
        )
    else:
        raise ValueError(f"unknown elicitation: {elicitation}")

    return f"{setup}\n\n{obs}\n\n{ask}"


# ---------------------------------------------------------------------------
# Parsing. Both parsers return None on failure; the unparsed rate is reported
# rather than silently dropped, because a high rate means the prompt is wrong.
# ---------------------------------------------------------------------------
def parse_flip(text, heads="H", tails="T"):
    if text is None:
        return None
    m = re.search(rf"[{heads}{tails}]", text.strip().upper())
    return None if m is None else int(m.group(0) == heads.upper())


def parse_probability(text):
    if text is None:
        return None
    m = re.search(r"(\d*\.?\d+)\s*(%?)", text.strip())
    if m is None:
        return None
    v = float(m.group(1))
    if m.group(2) == "%" or v > 1.0:
        v /= 100.0
    return v if 0.0 <= v <= 1.0 else None


# ---------------------------------------------------------------------------
# The single wiring point.
# ---------------------------------------------------------------------------
def call_model(prompt, model, n_samples=1, temperature=1.0, max_tokens=8):
    """Return `n_samples` raw completion strings. WIRE A REAL MODEL HERE.

    Kept provider-agnostic and lazily imported so the rest of this file runs on
    numpy alone. Neither branch is exercised by --selfcheck or --dry-run.
    """
    if model.startswith("claude"):
        try:
            import anthropic
        except ImportError:
            raise NotImplementedError(
                "pip install anthropic, and set ANTHROPIC_API_KEY in your shell"
            )
        client = anthropic.Anthropic()
        out = []
        for _ in range(n_samples):
            r = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            out.append(r.content[0].text if r.content else "")
        return out

    if model.startswith(("gpt", "o1", "o3", "o4")):
        try:
            import openai
        except ImportError:
            raise NotImplementedError(
                "pip install openai, and set OPENAI_API_KEY in your shell"
            )
        client = openai.OpenAI()
        r = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            n=n_samples, messages=[{"role": "user", "content": prompt}],
        )
        return [c.message.content or "" for c in r.choices]

    raise NotImplementedError(f"no client wired for model {model!r}")


def mock_model(path_name, rng, heads="H", tails="T"):
    """A stand-in that emits real-format strings drawn from a known path.

    `collapsed` is the mode-collapse failure: it answers its argmax every time
    instead of sampling, which is the case the null cells exist to catch.
    """
    def _call(prompt, model, n_samples=1, temperature=1.0, max_tokens=8, cell=None,
              elicitation="sampled"):
        p = PATHS[path_name](cell["d"], cell["n"], cell["s"])
        if elicitation in ("verbalized", "control"):
            v = typed_posterior_high(cell["d"], cell["n"], cell["s"]) \
                if elicitation == "control" else p
            return [f"{v:.3f}"] * n_samples
        if model == "mock-collapsed":
            return [(heads if p >= 0.5 else tails)] * n_samples
        draws = rng.random(n_samples) < p
        return [heads if y else tails for y in draws]
    return _call


# ---------------------------------------------------------------------------
# Running. One JSONL record per call, so re-analysis is free and a run resumes.
# ---------------------------------------------------------------------------
def run(cells, model, k, out_path, caller, conditions=CONDITIONS, verbose=True):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["cell"], r["condition"], r["elicitation"], r["rep"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_calls = 0
    with open(out_path, "a") as fh:
        for cell in cells:
            for condition in conditions:
                jobs = []
                if cell["family"] in SAMPLED_FAMILIES:
                    jobs.append(("sampled", k))
                jobs.append(("verbalized", 1))
                if condition == "known":
                    jobs.append(("control", 1))

                for elicitation, reps in jobs:
                    todo = [i for i in range(reps)
                            if (cell["id"], condition, elicitation, i) not in done]
                    if not todo:
                        continue
                    prompt = build_prompt(cell, condition, elicitation)
                    kwargs = {}
                    if caller is not call_model:
                        kwargs = {"cell": cell, "elicitation": elicitation}
                    texts = caller(prompt, model, n_samples=len(todo), **kwargs)
                    n_calls += len(todo)
                    for i, text in zip(todo, texts):
                        value = (parse_flip(text) if elicitation == "sampled"
                                 else parse_probability(text))
                        fh.write(json.dumps({
                            "cell": cell["id"], "family": cell["family"],
                            "d": cell["d"], "n": cell["n"], "s": cell["s"],
                            "condition": condition, "elicitation": elicitation,
                            "rep": i, "model": model,
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "prompt_len": len(prompt), "raw": text, "value": value,
                        }) + "\n")
                    fh.flush()
                if verbose:
                    print(f"    {cell['id']:<28} {condition:<8} done")
    return n_calls, out_path


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------
def _binom_ll(heads, k, p):
    """Log-likelihood of `heads` successes in k, up to the constant coefficient."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return heads * np.log(p) + (k - heads) * np.log(1 - p)


def _chi2_quantile(df, z=Z_99):
    """Wilson-Hilferty approximation. Adequate here; df is small and fixed."""
    return df * (1 - 2 / (9 * df) + z * np.sqrt(2 / (9 * df))) ** 3


def score(records, cells, condition="known", verbose=True):
    by_id = {c["id"]: c for c in cells}
    agg = {}
    for r in records:
        if r["condition"] != condition or r["elicitation"] != "sampled":
            continue
        if r["value"] is None:
            agg.setdefault(r["cell"], [0, 0, 0])[2] += 1
            continue
        a = agg.setdefault(r["cell"], [0, 0, 0])
        a[0] += r["value"]
        a[1] += 1

    if not agg:
        return None

    # Mode-collapse detector: on the null cells all three paths predict 0.5, so
    # a sampling model lands near 0.5 and an argmax-answering one does not.
    nulls = []
    for cid, (heads, k, _) in agg.items():
        if by_id[cid]["family"] == "null" and k:
            nulls.append((cid, heads / k))
    collapsed = [(c, p) for c, p in nulls
                 if p < MODE_COLLAPSE_CUT or p > 1 - MODE_COLLAPSE_CUT]

    rows, totals = [], {name: 0.0 for name in PATHS}
    ll_saturated, df = 0.0, 0
    for cid, (heads, k, bad) in sorted(agg.items()):
        cell = by_id[cid]
        if k == 0:
            continue
        p_hat = heads / k
        lls = {name: _binom_ll(heads, k, cell[name]) for name in PATHS}
        if cell["family"] != "null":          # null cells cannot discriminate
            for name in PATHS:
                totals[name] += lls[name]
            ll_saturated += _binom_ll(heads, k, p_hat)
            df += 1
        rows.append((cid, cell, p_hat, k, bad, lls))

    best = max(totals, key=totals.get)
    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    llr = ordered[0][1] - ordered[1][1]
    g2 = 2 * (ll_saturated - totals[best])
    cut = _chi2_quantile(max(df, 1))
    verdict = "none" if g2 > cut else best

    if verbose:
        se = lambda p, k: np.sqrt(max(p * (1 - p), 1e-12) / k)
        print(f"\n  condition = {condition}")
        print(f"  {'cell':<28}{'p_hat':>8}{'95% CI':>18}{'typed':>8}"
              f"{'flat':>7}{'lap':>8}   best")
        for cid, cell, p_hat, k, bad, lls in rows:
            h = 1.96 * se(p_hat, k)
            mark = "  (null)" if cell["family"] == "null" else ""
            b = max(lls, key=lls.get)
            print(f"  {cid:<28}{p_hat:8.3f}  [{max(0,p_hat-h):.3f}, "
                  f"{min(1,p_hat+h):.3f}]{cell['typed']:8.3f}{cell['flat']:7.3f}"
                  f"{cell['laplace']:8.3f}   {b}{mark}"
                  + (f"  [{bad} unparsed]" if bad else ""))
        print(f"\n  total log-likelihood over {df} discriminating cells:")
        for name, v in ordered:
            print(f"    {name:<10}{v:12.2f}")
        print(f"  log-likelihood ratio, best over runner-up: {llr:.2f} "
              f"({'decisive' if llr >= LLR_DECISIVE else 'not decisive'} "
              f"at the registered cut of {LLR_DECISIVE:.0f})")
        print(f"  fit of the winner: G2 = {g2:.2f} against a 1% cut of {cut:.2f}"
              f"  -> {'no path fits' if verdict == 'none' else 'winner fits'}")
        if collapsed:
            print("  MODE COLLAPSE DETECTED on null cells "
                  + ", ".join(f"{c} p_hat={p:.3f}" for c, p in collapsed))
            print("  The sampled channel is measuring an argmax, not a forecast. "
                  "Read the verbalized channel as primary.")
        print(f"\n  VERDICT: {verdict}")

    return {"verdict": verdict, "best": best, "llr": llr, "totals": totals,
            "g2": g2, "g2_cut": cut, "df": df, "mode_collapse": collapsed}


def report_verbalized(records, cells, verbose=True):
    """The secondary channel, including the paper's E5 slope."""
    by_id = {c["id"]: c for c in cells}
    vals = {}
    for r in records:
        if r["elicitation"] == "verbalized" and r["value"] is not None:
            vals[(r["cell"], r["condition"])] = r["value"]
    if not vals:
        return None

    if verbose:
        print("\n  verbalized channel (one call per cell, no sampling noise)")
        print(f"  {'cell':<28}{'cond':<9}{'stated':>8}{'typed':>8}{'flat':>7}{'lap':>8}")
        for (cid, cond), v in sorted(vals.items()):
            c = by_id[cid]
            print(f"  {cid:<28}{cond:<9}{v:8.3f}{c['typed']:8.3f}"
                  f"{c['flat']:7.3f}{c['laplace']:8.3f}")

    slope = {}
    for cond in CONDITIONS:
        xs, ys = [], []
        for c in cells:
            if c["family"] == "slope" and (c["id"], cond) in vals:
                xs.append(c["typed"] - 0.5)
                ys.append(vals[(c["id"], cond)] - 0.5)
        if len(xs) >= 2 and np.ptp(xs) > 0:
            slope[cond] = float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
    if verbose and slope:
        print("\n  E5 slope over the paper's one-head sweep "
              "(1 = typed, 0 = frozen; three points, no error bar):")
        for cond, sl in slope.items():
            print(f"    {cond:<9}{sl:8.3f}")
    return {"values": vals, "slope": slope}


# ---------------------------------------------------------------------------
# Self-check. Every number the design rests on, verified against the theory.
# ---------------------------------------------------------------------------
def selfcheck():
    print("Reference paths against coin_example.TypedReasoner and closed forms:")

    for d in (0.05, 0.10, 0.20, 0.30, 0.40):
        got, want = typed_path(d, 1, 1), 0.5 + 2 * d ** 2
        assert abs(got - want) < 1e-12, (d, got, want)
    print("    typed_path(d, 1, 1) == 0.5 + 2d^2                        OK")

    for d, want in ((0.05, 0.505), (0.10, 0.520), (0.30, 0.680)):
        assert abs(typed_path(d, 1, 1) - want) < 1e-12, (d, want)
    print("    the paper's sweep reproduces 0.505, 0.520, 0.680         OK")

    assert abs(typed_path(0.1, 1, 1) - (0.5 + 0.01 / 0.5)) < 1e-12
    print("    d=0.1 one head == m + v/m == 0.52                        OK")

    for d in (0.05, 0.20, 0.40):
        for n in (1, 5, 10, 20):
            for s in range(n + 1):
                p = typed_path(d, n, s)
                assert 0.5 - d - 1e-12 <= p <= 0.5 + d + 1e-12, (d, n, s, p)
    print("    typed_path stays inside (0.5-d, 0.5+d): it saturates     OK")

    assert abs(laplace_path(0.2, 20, 20) - 21 / 22) < 1e-12
    assert abs(laplace_path(0.2, 10, 5) - 0.5) < 1e-12
    print("    laplace_path == (s+1)/(n+2), unbounded                   OK")

    assert abs(typed_posterior_high(0.2, 20, 20) - 1.0) < 1e-3
    assert abs(typed_posterior_high(0.2, 10, 5) - 0.5) < 1e-12
    print("    typed_posterior_high, the arithmetic control's answer    OK")

    assert required_k(0.5, 0.5) == float("inf")
    for (d, n, s), want in (((0.05, 1, 1), 78487), ((0.10, 1, 1), 4904),
                            ((0.20, 20, 20), 47), ((0.40, 10, 7), 10)):
        got = required_k(typed_path(d, n, s), 0.5)
        assert got == want, ((d, n, s), got, want)
    print("    required_k reproduces the design table                   OK")

    cells = select_grid()
    print(f"\nGrid: {len(cells)} cells, k = {K_DEFAULT} in the sampled channel.")
    print(f"  {'cell':<28}{'typed':>8}{'flat':>7}{'lap':>8}"
          f"{'|T-F|':>8}{'|T-L|':>8}{'k:TvF':>9}{'k:TvL':>9}")
    for c in cells:
        f = lambda v: "  inf" if v == float("inf") else f"{v:5.0f}"
        print(f"  {c['id']:<28}{c['typed']:8.3f}{c['flat']:7.3f}{c['laplace']:8.3f}"
              f"{c['gap_tf']:8.3f}{c['gap_tl']:8.3f}{f(c['k_tf']):>9}{f(c['k_tl']):>9}")

    disc = [c for c in cells
            if c["family"] in SAMPLED_FAMILIES and c["family"] != "null"]
    for c in disc:
        if c["family"] == "paper":
            assert c["k_tf"] <= K_DEFAULT, c        # typed-vs-flat only, by design
        else:
            assert max(c["k_tf"], c["k_tl"]) <= K_DEFAULT, c
    print(f"\n    every discriminating cell is resolvable at k = {K_DEFAULT}  OK")

    for c in cells:
        if c["family"] == "null":
            assert max(c["gap_tf"], c["gap_tl"], c["gap_fl"]) < 0.02, c
    print("    null cells: all three paths agree, so they discriminate")
    print("    nothing and serve as the mode-collapse detector          OK")

    paper = [c for c in cells if c["family"] == "paper" and c["s"] == 1][0]
    assert paper["k_tl"] > 1000, paper
    print("    the paper's d=0.30 one-head cell cannot separate typed")
    print(f"    from counting (needs k = {paper['k_tl']:.0f})              OK")

    n_sampled = sum(1 for c in cells if c["family"] in SAMPLED_FAMILIES)
    calls = n_sampled * K_DEFAULT * len(CONDITIONS)
    extra = len(cells) * len(CONDITIONS) + len(cells)
    print(f"\nBudget: {n_sampled} sampled cells x {K_DEFAULT} x {len(CONDITIONS)} "
          f"conditions = {calls} single-token calls,")
    print(f"        plus {extra} verbalized and control calls. Total {calls + extra}.")

    p = build_prompt(cells[0], "known", "sampled")
    assert "50% of the time" not in p and "30%" in p and "70%" in p, p
    assert parse_flip("H") == 1 and parse_flip(" t ") == 0 and parse_flip("x") is None
    assert parse_probability("0.52") == 0.52 and abs(parse_probability("52%") - 0.52) < 1e-9
    assert parse_probability("nope") is None
    print("    prompt builder and both parsers                          OK")

    print("\nExample prompt (saturation cell, known mixture, sampled):")
    print("    " + p.replace("\n", "\n    "))
    print("\nAll self-checks passed.")


def dry_run(path_name, k, seed=20260810):
    """Drive the full pipeline against a seeded mock and score the result."""
    cells = select_grid()
    rng = np.random.default_rng(seed)
    caller = mock_model("typed" if path_name == "collapsed" else path_name, rng)
    model = "mock-collapsed" if path_name == "collapsed" else f"mock-{path_name}"
    out = os.path.join(RESULTS_DIR, f"dryrun_{path_name}.jsonl")
    if os.path.exists(out):
        os.remove(out)

    print(f"Dry run: mock '{path_name}', k = {k}, no API calls.")
    n_calls, _ = run(cells, model, k, out, caller, verbose=False)
    records = [json.loads(l) for l in open(out)]
    print(f"  {n_calls} mock calls, {len(records)} records -> {os.path.relpath(out)}")

    res = score(records, cells, condition="known")
    report_verbalized(records, cells)
    return res, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", choices=["typed", "flat", "laplace", "collapsed"])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", metavar="JSONL")
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return 0

    if args.dry_run:
        if not args.mock:
            ap.error("--dry-run needs --mock {typed,flat,laplace,collapsed}")
        res, _ = dry_run(args.mock, args.k)
        # The collapsed mock answers its argmax instead of sampling. The right
        # result is a refusal to name a path: on its own the likelihood scorer
        # calls it 'laplace' decisively, and the two guards (the G2 fit test and
        # the null-cell detector) are what turn that into 'none'.
        expect = "none" if args.mock == "collapsed" else args.mock
        ok = res["verdict"] == expect
        print(f"\n  scorer returned '{res['verdict']}', expected '{expect}' for "
              f"mock '{args.mock}' -> {'OK' if ok else 'MISMATCH'}")
        if args.mock == "collapsed" and ok:
            assert res["mode_collapse"], "detector should have fired"
            print("  (and the mode-collapse detector fired, as it must)")
        return 0 if ok else 1

    if args.analyze:
        cells = select_grid()
        records = [json.loads(l) for l in open(args.analyze)]
        for cond in CONDITIONS:
            score(records, cells, condition=cond)
        report_verbalized(records, cells)
        return 0

    if args.run:
        if not args.model:
            ap.error("--run needs --model, e.g. --model claude-haiku-4-5-20251001")
        cells = select_grid()
        stamp = time.strftime("%Y%m%dT%H%M%S")
        out = os.path.join(RESULTS_DIR, f"freeze_{args.model}_{stamp}.jsonl")
        print(f"Running E-A on {args.model}, k = {args.k}. Appending to "
              f"{os.path.relpath(out)}")
        n_calls, _ = run(cells, args.model, args.k, out, call_model)
        records = [json.loads(l) for l in open(out)]
        print(f"\n{n_calls} calls made.")
        for cond in CONDITIONS:
            score(records, cells, condition=cond)
        report_verbalized(records, cells)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
