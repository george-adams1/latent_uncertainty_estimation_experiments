"""Analyze E-C records with item-level bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import json
import sys

from eb.analyze import bootstrap_ci, pearson


METRICS = {
    "observed_ambiguous_variance": lambda row: row["estimator"]["observed_ambiguous_variance"],
    "within_reading_variance": lambda row: row["estimator"]["within_reading_variance"],
    "between_reading_variance": lambda row: row["estimator"]["between_reading_variance"],
    "between_fraction": lambda row: row["estimator"]["between_fraction"],
    "realized_gain": lambda row: row["realized_gain"],
}


def load_records(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(rows: list[dict], value_fn) -> float:
    return sum(value_fn(row) for row in rows) / len(rows) if rows else float("nan")


def _corr(rows: list[dict], x_fn, y_fn=lambda row: row["realized_gain"]) -> float:
    return pearson([x_fn(row) for row in rows], [y_fn(row) for row in rows])


def analyze(
    records: list[dict],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict:
    set_a = [row for row in records if row["set"] == "A"]
    set_b = [row for row in records if row["set"] == "B"]
    if not set_a or not set_b:
        raise ValueError("E-C analysis requires nonempty Set A and Set B records")

    result = {
        "n_setA": len(set_a),
        "n_setB": len(set_b),
        "means": {"setA": {}, "setB": {}},
        "confidence_intervals": {
            "level": confidence_level,
            "method": "item-level percentile bootstrap; Set A/Set B stratified for contrasts and pooled correlations",
            "bootstrap_samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "means": {"setA": {}, "setB": {}},
            "contrasts": {},
            "correlations": {},
            "correlation_contrasts": {},
        },
    }
    ci = result["confidence_intervals"]
    for set_name, rows in (("setA", set_a), ("setB", set_b)):
        for metric_name, value_fn in METRICS.items():
            result["means"][set_name][metric_name] = _mean(rows, value_fn)
            ci["means"][set_name][metric_name] = bootstrap_ci(
                [rows],
                lambda sampled, fn=value_fn: _mean(sampled, fn),
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
                label=f"mean:{set_name}:{metric_name}",
            )

    result["contrasts"] = {}
    for metric_name, value_fn in METRICS.items():
        point = _mean(set_a, value_fn) - _mean(set_b, value_fn)
        result["contrasts"][f"setA_minus_setB_{metric_name}"] = point
        ci["contrasts"][f"setA_minus_setB_{metric_name}"] = bootstrap_ci(
            [set_a, set_b],
            lambda a, b, fn=value_fn: _mean(a, fn) - _mean(b, fn),
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            label=f"contrast:{metric_name}",
        )

    correlation_specs = {
        "setA_between_variance_vs_gain": (
            [set_a],
            lambda a: _corr(a, METRICS["between_reading_variance"]),
        ),
        "setA_observed_variance_vs_gain": (
            [set_a],
            lambda a: _corr(a, METRICS["observed_ambiguous_variance"]),
        ),
        "setA_confidence_vs_gain": (
            [set_a],
            lambda a: _corr(a, lambda row: row["confidence"]),
        ),
        "pooled_between_variance_vs_gain": (
            [set_a, set_b],
            lambda a, b: _corr(a + b, METRICS["between_reading_variance"]),
        ),
        "pooled_observed_variance_vs_gain": (
            [set_a, set_b],
            lambda a, b: _corr(a + b, METRICS["observed_ambiguous_variance"]),
        ),
        "pooled_confidence_vs_gain": (
            [set_a, set_b],
            lambda a, b: _corr(a + b, lambda row: row["confidence"]),
        ),
    }
    result["correlations"] = {}
    for name, (groups, statistic) in correlation_specs.items():
        value = statistic(*groups)
        result["correlations"][name] = value
        ci["correlations"][name] = bootstrap_ci(
            groups,
            statistic,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            label=f"correlation:{name}",
        )

    correlation_contrast_specs = {
        "setA_between_minus_observed_variance_gain_correlation": (
            [set_a],
            lambda a: _corr(a, METRICS["between_reading_variance"])
            - _corr(a, METRICS["observed_ambiguous_variance"]),
        ),
        "setA_between_minus_confidence_gain_correlation": (
            [set_a],
            lambda a: _corr(a, METRICS["between_reading_variance"])
            - _corr(a, lambda row: row["confidence"]),
        ),
        "pooled_between_minus_observed_variance_gain_correlation": (
            [set_a, set_b],
            lambda a, b: _corr(a + b, METRICS["between_reading_variance"])
            - _corr(a + b, METRICS["observed_ambiguous_variance"]),
        ),
        "pooled_between_minus_confidence_gain_correlation": (
            [set_a, set_b],
            lambda a, b: _corr(a + b, METRICS["between_reading_variance"])
            - _corr(a + b, lambda row: row["confidence"]),
        ),
    }
    result["correlation_contrasts"] = {}
    for name, (groups, statistic) in correlation_contrast_specs.items():
        result["correlation_contrasts"][name] = statistic(*groups)
        ci["correlation_contrasts"][name] = bootstrap_ci(
            groups,
            statistic,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            label=f"correlation_contrast:{name}",
        )

    result["max_abs_decomposition_identity_error"] = max(
        abs(row["estimator"]["identity_error"]) for row in records
    )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args(argv)
    result = analyze(
        load_records(args.results_path),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    json.dump(result, sys.stdout, indent=2, allow_nan=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
