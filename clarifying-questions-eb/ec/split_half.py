"""Split-half re-estimate of the E-C between-variance/gain correlation.

The headline E-C correlation estimates the predictor (between-reading variance)
and the outcome (realized clarification gain) from the same 32 samples per
fixed-reading prompt, so shared sampling noise inflates it. This recomputes it
with the two quantities drawn from disjoint halves of the existing samples: no
new inference, at the cost of estimating each from 16 draws instead of 32.

Both assignments are reported (variance from the first half and gain from the
second, then the mirror) because neither half is privileged.

    python3 -m ec.split_half scan_results/ec_qwen3_8b_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

from eb.analyze import bootstrap_ci, pearson

from .estimator import decompose_conditionals, empirical_distribution
from .run_experiment import strict_accuracy

HALF = 16


def _readings(record: dict) -> dict[str, list[str]]:
    return {name: data["answer_aliases"] for name, data in record["readings"].items()}


def _between(record: dict, lo: int, hi: int) -> float:
    readings = _readings(record)
    categories = list(readings) + ["multiple", "other"]
    conditionals = {
        name: empirical_distribution(data["prompt"]["clusters"][lo:hi], categories)
        for name, data in record["readings"].items()
    }
    return decompose_conditionals(conditionals)["between_reading_variance"]


def _gain(record: dict, lo: int, hi: int) -> float:
    """Gain with clarified accuracy from one half; the baseline is untouched.

    The baseline comes from the ambiguous prompt, which never enters the
    variance estimate, so it needs no splitting to decouple the two.
    """
    readings = _readings(record)
    ambiguous = record["ambiguous_prompt"]["samples"]
    gains = []
    for name, data in record["readings"].items():
        clarified = strict_accuracy(data["prompt"]["samples"][lo:hi], name, readings)
        baseline = strict_accuracy(ambiguous, name, readings)
        gains.append(clarified - baseline)
    return sum(gains) / len(gains)


def _corr(rows: list[dict], x_fn, y_fn, label: str) -> tuple[float, dict]:
    """Materialize each item's (x, y) once; the bootstrap then only resamples."""
    pairs = [{"x": x_fn(r), "y": y_fn(r)} for r in rows]
    point = pearson([p["x"] for p in pairs], [p["y"] for p in pairs])
    ci = bootstrap_ci(
        [pairs],
        lambda group: pearson([p["x"] for p in group], [p["y"] for p in group]),
        label=label,
    )
    return point, ci


def analyze(path: str) -> dict:
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    set_a = [r for r in rows if r["set"] == "A"]

    same, same_ci = _corr(
        set_a,
        lambda r: r["estimator"]["between_reading_variance"],
        lambda r: r["realized_gain"],
        "same_samples",
    )
    first, first_ci = _corr(
        set_a,
        lambda r: _between(r, 0, HALF),
        lambda r: _gain(r, HALF, 2 * HALF),
        "split_first",
    )
    second, second_ci = _corr(
        set_a,
        lambda r: _between(r, HALF, 2 * HALF),
        lambda r: _gain(r, 0, HALF),
        "split_second",
    )
    return {
        "path": path,
        "n_setA": len(set_a),
        "same_samples": {"r": same, **same_ci},
        "split_first_half_predictor": {"r": first, **first_ci},
        "split_second_half_predictor": {"r": second, **second_ci},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args(argv)
    for path in args.results:
        out = analyze(path)
        print(f"\n{out['path']}  (n = {out['n_setA']} Set A items)")
        for key in ("same_samples", "split_first_half_predictor", "split_second_half_predictor"):
            row = out[key]
            print(f"  {key:32s} r = {row['r']:+.3f}   95% CI [{row['low']:+.3f}, {row['high']:+.3f}]")


if __name__ == "__main__":
    sys.exit(main())
