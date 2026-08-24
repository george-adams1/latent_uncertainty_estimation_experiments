"""Categorical variance decomposition used by E-C.

Represent an answer cluster as a one-hot random vector X. Its total variance
(the trace of the covariance matrix) is ``1 - sum_c p(c)^2``, also known as
Gini impurity or the probability that two independent draws disagree.

For a uniformly sampled intended reading W, the law of total variance gives

    Var(X) = E_W[Var(X | W)] + Var_W(E[X | W]).

The first term is within-reading answer variability. The second is the
between-reading component that a clarification can, in principle, remove.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def empirical_distribution(labels: Sequence[str], categories: Sequence[str] | None = None) -> dict[str, float]:
    """Return a normalized categorical distribution, retaining zero categories."""
    if not labels:
        raise ValueError("at least one label is required")
    ordered = list(dict.fromkeys(categories or labels))
    unseen = [label for label in labels if label not in ordered]
    ordered.extend(dict.fromkeys(unseen))
    total = len(labels)
    return {category: labels.count(category) / total for category in ordered}


def categorical_variance(distribution: Mapping[str, float]) -> float:
    """Trace variance of a one-hot categorical variable: 1 - ||p||^2."""
    total = sum(distribution.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"distribution must sum to one, got {total}")
    if any(probability < 0.0 for probability in distribution.values()):
        raise ValueError("distribution probabilities must be nonnegative")
    return 1.0 - sum(probability * probability for probability in distribution.values())


def decompose_conditionals(
    conditionals: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float] | None = None,
) -> dict:
    """Apply total-variance decomposition to reading-conditioned distributions.

    ``conditionals`` maps reading name to an answer-cluster distribution.
    When ``weights`` is omitted, readings receive equal probability, matching
    E-C's design prior over intended readings.
    """
    if not conditionals:
        raise ValueError("at least one reading-conditioned distribution is required")
    reading_names = list(conditionals)
    if weights is None:
        weights = {name: 1.0 / len(reading_names) for name in reading_names}
    if set(weights) != set(reading_names):
        raise ValueError("weights must name exactly the supplied readings")
    weight_total = sum(weights.values())
    if not math.isclose(weight_total, 1.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to one, got {weight_total}")
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("weights must be nonnegative")

    categories = sorted({category for distribution in conditionals.values() for category in distribution})
    normalized: dict[str, dict[str, float]] = {}
    for name, distribution in conditionals.items():
        expanded = {category: float(distribution.get(category, 0.0)) for category in categories}
        # Validate before using the distribution in the mixture.
        categorical_variance(expanded)
        normalized[name] = expanded

    mixture = {
        category: sum(weights[name] * normalized[name][category] for name in reading_names)
        for category in categories
    }
    total = categorical_variance(mixture)
    within = sum(weights[name] * categorical_variance(normalized[name]) for name in reading_names)
    # The algebraic identity is exact; clamp only floating-point dust.
    between = total - within
    if between < 0.0 and between > -1e-12:
        between = 0.0
    if between < 0.0:
        raise ValueError(f"negative between-reading variance: {between}")
    direct_between = sum(
        weights[name]
        * sum((normalized[name][category] - mixture[category]) ** 2 for category in categories)
        for name in reading_names
    )
    return {
        "weights": dict(weights),
        "mixture_distribution": mixture,
        "total_variance": total,
        "within_reading_variance": within,
        "between_reading_variance": between,
        "between_reading_variance_direct": direct_between,
        "identity_error": total - within - direct_between,
        "between_fraction": between / total if total > 0.0 else 0.0,
    }


def total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Total-variation distance between categorical distributions."""
    categories = set(left) | set(right)
    return 0.5 * sum(abs(left.get(category, 0.0) - right.get(category, 0.0)) for category in categories)

