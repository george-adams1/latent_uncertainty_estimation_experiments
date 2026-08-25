"""Summaries and split-half correlations for the E-C clustering arm."""
from __future__ import annotations

from collections import Counter

from eb.analyze import bootstrap_ci, pearson


ARMS = ("oracle", "q", "qs", "answer_string")


def _corr(rows: list[dict], x_key: str, y_key: str, label: str, **bootstrap) -> dict:
    pairs = [
        {
            "x": row["estimates"][x_key][y_key]["between_reading_variance"],
            "y": row["realized_gain"][
                "split_second" if y_key == "split_first" else "split_first" if y_key == "split_second" else "full"
            ],
        }
        for row in rows
    ]
    point = pearson([pair["x"] for pair in pairs], [pair["y"] for pair in pairs])
    ci = bootstrap_ci(
        [pairs],
        lambda sample: pearson([pair["x"] for pair in sample], [pair["y"] for pair in sample]),
        label=label,
        **bootstrap,
    )
    return {"r": point, **ci}


def _count_distribution(rows: list[dict], variant: str) -> dict[str, int]:
    counts = Counter(len(row["listers"][variant]["interpretations"]) for row in rows)
    return {str(count): counts.get(count, 0) for count in range(1, 5)}


def _aggregate_audit(rows: list[dict], variant: str) -> dict:
    audits = [row["set_a_agreement_audit"][variant] for row in rows]
    matches_all = sum(row["matches_all"] for row in audits)
    n_all = sum(row["n_all"] for row in audits)
    matches_unique = sum(row["matches_unique_oracle"] for row in audits)
    n_unique = sum(row["n_unique_oracle"] for row in audits)
    recovered = sum(row["oracle_reading_recall"] == 1.0 for row in audits)
    return {
        "alignment_method": audits[0]["method"] if audits else None,
        "matches_all": matches_all,
        "n_all": n_all,
        "agreement_all": matches_all / n_all if n_all else None,
        "matches_unique_oracle": matches_unique,
        "n_unique_oracle": n_unique,
        "agreement_unique_oracle": matches_unique / n_unique if n_unique else None,
        "items_recovering_all_oracle_readings": recovered,
        "n_items": len(audits),
        "item_recall_rate": recovered / len(audits) if audits else None,
        "none_count": sum(row["none_count"] for row in audits),
    }


def _batch_mean(rows: list[dict], arm: str) -> float:
    if not rows:
        return float("nan")
    return sum(
        row["set_b_batch_diagnostic"]["arms"][arm]["full"]["between_reading_variance"]
        for row in rows
    ) / len(rows)


def analyze_clustering(
    records: list[dict],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict:
    set_a = [row for row in records if row["set"] == "A"]
    set_b = [row for row in records if row["set"] == "B"]
    bootstrap = {
        "samples": bootstrap_samples,
        "seed": bootstrap_seed,
        "confidence_level": confidence_level,
    }
    correlations = {}
    for arm in ARMS:
        correlations[arm] = {
            "same_samples": _corr(set_a, arm, "full", f"clustering:{arm}:same", **bootstrap),
            "split_first_half_predictor": _corr(
                set_a, arm, "split_first", f"clustering:{arm}:first", **bootstrap
            ),
            "split_second_half_predictor": _corr(
                set_a, arm, "split_second", f"clustering:{arm}:second", **bootstrap
            ),
        }

    cluster_counts = {
        variant: {
            "setA": _count_distribution(set_a, variant),
            "setB": _count_distribution(set_b, variant),
        }
        for variant in ("q", "qs")
    }
    none_counts = {}
    for variant in ("q", "qs"):
        none_counts[variant] = {}
        for name, rows in (("setA", set_a), ("setB", set_b)):
            labels = [
                assignment["label"]
                for row in rows
                for batch in row["assignments"][variant].values()
                for assignment in batch
            ]
            count = labels.count("none")
            none_counts[variant][name] = {
                "count": count,
                "n": len(labels),
                "rate": count / len(labels) if labels else None,
            }

    special = [
        row["set_a_agreement_audit"]["oracle_special_counts"] for row in set_a
    ]
    batch_means = {arm: _batch_mean(set_b, arm) for arm in ARMS}
    batch_intervals = {
        arm: bootstrap_ci(
            [set_b],
            lambda sampled, selected=arm: _batch_mean(sampled, selected),
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            label=f"clustering:setB_between_batch:{arm}",
        )
        for arm in ARMS
    }
    summary = {
        "schema_version": "ec.clustering.summary.v2",
        "pilot": all(row["pilot"] for row in records),
        "analysis_status": (
            "pilot_only_not_confirmatory" if all(row["pilot"] for row in records) else "frozen_full_run"
        ),
        "subject_model": records[0]["subject_model"] if records else None,
        "clusterer": records[0]["clusterer"] if records else None,
        "source": records[0]["source"] if records else None,
        "n_setA": len(set_a),
        "n_setB": len(set_b),
        "confidence_intervals": {
            "method": "item-level percentile bootstrap, identical settings to E-C split-half",
            "level": confidence_level,
            "bootstrap_samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "split_half_correlations": correlations,
        "lister_cluster_count_distributions": cluster_counts,
        "setA_agreement_audit": {
            variant: _aggregate_audit(set_a, variant) for variant in ("q", "qs")
        },
        "none_assignments": none_counts,
        "setA_oracle_special_counts": {
            "multiple": sum(row["multiple"] for row in special),
            "other": sum(row["other"] for row in special),
            "n": sum(row["n"] for row in special),
        },
        "setB_between_batch_diagnostic": {
            "name": "between_batch_variance",
            "interpretation": (
                "finite-sample separation between original and repeat batches; "
                "not between-reading variance"
            ),
            "mean_by_arm": batch_means,
            "confidence_intervals": batch_intervals,
        },
        "mean_between_variance": {
            arm: {
                "setA": sum(row["estimates"][arm]["full"]["between_reading_variance"] for row in set_a) / len(set_a),
                "setB": sum(row["estimates"][arm]["full"]["between_reading_variance"] for row in set_b) / len(set_b),
            }
            for arm in ARMS
        },
        "mean_observed_categorical_variance": {
            arm: {
                "setA": sum(row["estimates"][arm]["observed_categorical_variance"] for row in set_a) / len(set_a),
                "setB": sum(row["estimates"][arm]["observed_categorical_variance"] for row in set_b) / len(set_b),
            }
            for arm in ARMS
        },
    }
    return summary
