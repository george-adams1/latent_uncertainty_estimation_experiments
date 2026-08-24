"""The three predictions from experiment-matched-confidence.md:

1. Gain from asking is large on Set A, near zero on Set B.
2. Pre-ask confidence does not predict the gain (Corollary 1 in observable form).
3. The cluster diagnostic does predict the gain.

Usage: python -m eb.analyze results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from statistics import NormalDist

CORRECT = {"correct": 1.0, "wrong": 0.0, "hedged": 0.0}  # strict grading


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def accuracy(records: list[dict], condition: str) -> float:
    if not records:
        return float("nan")
    return sum(CORRECT[r[condition]["grade"]] for r in records) / len(records)


def per_item_gain(records: list[dict], ask_condition: str = "self_ask") -> list[float]:
    return [CORRECT[r[ask_condition]["grade"]] - CORRECT[r["answer_now"]["grade"]] for r in records]


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator


def _percentile(values: list[float], q: float) -> float:
    """Linearly interpolated percentile for a sorted bootstrap distribution."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    groups: list[list[dict]],
    statistic,
    *,
    samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    label: str = "",
) -> dict:
    """Percentile bootstrap CI, resampling each supplied group independently.

    Supplying Set A and Set B separately gives a stratified bootstrap for pooled
    statistics while preserving each set's observed sample size. A stable label
    gives every reported statistic its own deterministic random stream.
    """
    if samples <= 0 or not 0.0 < confidence_level < 1.0 or any(not group for group in groups):
        return {"low": float("nan"), "high": float("nan"), "valid_resamples": 0}
    rng = random.Random(f"{seed}:{label}")
    distribution = []
    for _ in range(samples):
        resampled = [rng.choices(group, k=len(group)) for group in groups]
        value = statistic(*resampled)
        if math.isfinite(value):
            distribution.append(value)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "low": _percentile(distribution, alpha),
        "high": _percentile(distribution, 1.0 - alpha),
        "valid_resamples": len(distribution),
    }


def wilson_ci(successes: int, total: int, confidence_level: float = 0.95) -> dict:
    """Wilson score interval for a binomial proportion."""
    if total <= 0 or not 0.0 < confidence_level < 1.0:
        return {"low": float("nan"), "high": float("nan"), "successes": successes, "n": total}
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    observed = successes / total
    denominator = 1.0 + z * z / total
    center = (observed + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(observed * (1.0 - observed) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
        "successes": successes,
        "n": total,
    }


def analyze(
    records: list[dict],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict:
    setA = [r for r in records if r["set"] == "A"]
    setB = [r for r in records if r["set"] == "B"]

    result = {
        "n_setA": len(setA),
        "n_setB": len(setB),
        "setA_accuracy": {
            "answer_now": accuracy(setA, "answer_now"),
            "oracle_clarify": accuracy(setA, "oracle_clarify"),
            "self_ask": accuracy(setA, "self_ask"),
        },
        "setB_accuracy": {
            "answer_now": accuracy(setB, "answer_now"),
            "self_ask": accuracy(setB, "self_ask"),
        },
    }

    # Prediction 1: gain by set. Oracle gain is the primary/ceiling measure
    # (experiment-ask-protocol.md); self-ask gain is the realized competence.
    result["prediction_1_gain"] = {
        "setA_oracle_gain": result["setA_accuracy"]["oracle_clarify"] - result["setA_accuracy"]["answer_now"],
        "setA_selfask_gain": result["setA_accuracy"]["self_ask"] - result["setA_accuracy"]["answer_now"],
        "setB_selfask_gain": result["setB_accuracy"]["self_ask"] - result["setB_accuracy"]["answer_now"],
    }

    # Predictions 2 and 3 pool both sets, using self-ask gain per item
    # (oracle-clarify has no Set B analogue by construction).
    pooled = setA + setB
    gains = per_item_gain(pooled, "self_ask")
    confidences = [r["confidence"] for r in pooled]
    diag_entropy = [r["diagnostic"]["entropy"] for r in pooled]
    diag_second = [r["diagnostic"]["second_largest_frac"] for r in pooled]

    result["prediction_2_confidence_vs_gain_corr"] = pearson(confidences, gains)
    result["prediction_3_entropy_vs_gain_corr"] = pearson(diag_entropy, gains)
    result["prediction_3_second_largest_vs_gain_corr"] = pearson(diag_second, gains)

    # Hedge rate, recorded per experiment-ask-protocol.md's "third response category".
    hedge_count = sum(1 for r in setA if r["answer_now"]["grade"] == "hedged")
    result["setA_answer_now_hedge_rate"] = hedge_count / len(setA) if setA else float("nan")

    def gain(rows: list[dict], condition: str) -> float:
        values = per_item_gain(rows, condition)
        return sum(values) / len(values) if values else float("nan")

    def pooled_corr(a_rows: list[dict], b_rows: list[dict], x_fn) -> float:
        rows = a_rows + b_rows
        return pearson(
            [x_fn(row) for row in rows],
            per_item_gain(rows, "self_ask"),
        )

    ci = {
        "level": confidence_level,
        "method": "Wilson score for binomial proportions; item-level percentile bootstrap paired within-item for gains and stratified by set for pooled correlations",
        "bootstrap_samples": bootstrap_samples,
        "seed": bootstrap_seed,
        "setA_accuracy": {},
        "setB_accuracy": {},
        "prediction_1_gain": {},
    }
    for condition in ("answer_now", "oracle_clarify", "self_ask"):
        ci["setA_accuracy"][condition] = wilson_ci(
            int(sum(CORRECT[row[condition]["grade"]] for row in setA)),
            len(setA),
            confidence_level,
        )
    for condition in ("answer_now", "self_ask"):
        ci["setB_accuracy"][condition] = wilson_ci(
            int(sum(CORRECT[row[condition]["grade"]] for row in setB)),
            len(setB),
            confidence_level,
        )
    for name, rows, condition in (
        ("setA_oracle_gain", setA, "oracle_clarify"),
        ("setA_selfask_gain", setA, "self_ask"),
        ("setB_selfask_gain", setB, "self_ask"),
    ):
        ci["prediction_1_gain"][name] = bootstrap_ci(
            [rows],
            lambda sampled, c=condition: gain(sampled, c),
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            label=f"prediction_1_gain:{name}",
        )
    ci["prediction_2_confidence_vs_gain_corr"] = bootstrap_ci(
        [setA, setB],
        lambda a, b: pooled_corr(a, b, lambda row: row["confidence"]),
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        label="prediction_2_confidence_vs_gain_corr",
    )
    ci["prediction_3_entropy_vs_gain_corr"] = bootstrap_ci(
        [setA, setB],
        lambda a, b: pooled_corr(a, b, lambda row: row["diagnostic"]["entropy"]),
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        label="prediction_3_entropy_vs_gain_corr",
    )
    ci["prediction_3_second_largest_vs_gain_corr"] = bootstrap_ci(
        [setA, setB],
        lambda a, b: pooled_corr(a, b, lambda row: row["diagnostic"]["second_largest_frac"]),
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        label="prediction_3_second_largest_vs_gain_corr",
    )
    ci["setA_answer_now_hedge_rate"] = wilson_ci(
        hedge_count,
        len(setA),
        confidence_level,
    )
    result["confidence_intervals"] = ci

    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args(argv)
    records = load_records(args.results_path)
    result = analyze(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
