import json
import copy

import pytest

from ec.clustering import InferenceClusterer, run_clustering_item
from ec.clustering_analysis import analyze_clustering
from ec.run_experiment import run_set_a_item, run_set_b_item

from .fixtures import CHIPMUNKS, GEORGIA
from .test_ec import DeterministicECClient


class DeterministicStructuredClient:
    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, schema_name, schema, max_tokens):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema_name": schema_name,
                "schema": schema,
                "max_tokens": max_tokens,
            }
        )
        if schema_name.startswith("ec_lister"):
            if "capital of Georgia" in user:
                payload = {
                    "interpretations": [
                        "The capital of the country named Georgia",
                        "The capital of the U.S. state named Georgia",
                    ]
                }
            else:
                payload = {"interpretations": ["The person who created The Chipmunks"]}
        elif "Tbilisi" in user:
            payload = {"assignment": "I1"}
        elif "Atlanta" in user:
            payload = {"assignment": "I2"}
        elif "David Seville" in user:
            payload = {"assignment": "I1"}
        else:
            payload = {"assignment": "none"}
        return payload, json.dumps(payload)


def _provenance(subject_model="fixture"):
    return {
        "subject_model": subject_model,
        "clusterer": {
            "model": "fixture-clusterer",
            "temperature": 0.0,
            "seed": 0,
            "prompt_sha256": {},
            "prompt_commit": None,
            "prompts_frozen": False,
        },
        "source": {
            "results_path": "fixture.jsonl",
            "results_sha256": "fixture",
            "results_commit": "fixture",
        },
    }


def test_annotation_blind_clustering_preserves_oracle_estimator_and_labels_every_batch():
    source = run_set_a_item(
        DeterministicECClient(), GEORGIA, 55.0, samples_per_prompt=32, temperature=1.0
    )
    client = DeterministicStructuredClient()
    record = run_clustering_item(InferenceClusterer(client), source, _provenance(), pilot=True)

    assert record["schema_version"] == "ec.clustering.v2"
    assert record["analysis_status"] == "pilot_only_not_confirmatory"
    assert len(record["listers"]["q"]["interpretations"]) == 2
    assert len(record["listers"]["qs"]["interpretations"]) == 2
    for variant in ("q", "qs"):
        assert set(record["assignments"][variant]) == {
            "ambiguous",
            "condition:reading_a",
            "condition:reading_b",
        }
        assert all(len(batch) == 32 for batch in record["assignments"][variant].values())
        assert record["estimates"][variant]["full"]["between_reading_variance"] == pytest.approx(0.5)
        assert record["set_a_agreement_audit"][variant]["oracle_reading_recall"] == 1.0
    assert record["estimates"]["oracle"]["full"]["between_reading_variance"] == pytest.approx(
        source["estimator"]["between_reading_variance"]
    )

    # No clusterer request contains annotation-only fixed-reading question
    # strings. Naturally occurring sampled answers remain allowed inputs.
    joined = "\n".join(call["user"] for call in client.calls)
    assert GEORGIA.reading_a.question not in joined
    assert GEORGIA.reading_b.question not in joined


def test_set_b_remains_structurally_zero_but_reports_lister_and_string_spread():
    source = run_set_b_item(
        DeterministicECClient(), CHIPMUNKS, 55.0, samples_per_prompt=32, temperature=1.0
    )
    record = run_clustering_item(
        InferenceClusterer(DeterministicStructuredClient()), source, _provenance(), pilot=True
    )
    assert len(record["listers"]["q"]["interpretations"]) == 1
    for arm in ("oracle", "q", "qs", "answer_string"):
        assert record["estimates"][arm]["full"]["between_reading_variance"] == pytest.approx(0.0)
        assert record["set_b_batch_diagnostic"]["arms"][arm]["full"][
            "between_reading_variance"
        ] == pytest.approx(0.0)
    assert record["estimates"]["answer_string"]["observed_categorical_variance"] == pytest.approx(0.0)


def test_set_b_between_batch_diagnostic_detects_spurious_string_separation():
    source = run_set_b_item(
        DeterministicECClient(), CHIPMUNKS, 55.0, samples_per_prompt=32, temperature=1.0
    )
    source = copy.deepcopy(source)
    source["original_prompt"]["samples"] = ["first surface form"] * 32
    source["repeat_prompt"]["samples"] = ["second surface form"] * 32
    record = run_clustering_item(
        InferenceClusterer(DeterministicStructuredClient()), source, _provenance(), pilot=True
    )
    # The paper's one-reading E-C estimand is unchanged.
    assert record["estimates"]["answer_string"]["full"]["between_reading_variance"] == 0.0
    # The explicitly separate null diagnostic exposes batch-specific strings.
    assert record["set_b_batch_diagnostic"]["arms"]["answer_string"]["full"][
        "between_reading_variance"
    ] == pytest.approx(0.5)


def test_summary_contains_all_split_half_arms_and_pilot_marking():
    source_a = run_set_a_item(
        DeterministicECClient(), GEORGIA, 55.0, samples_per_prompt=32, temperature=1.0
    )
    source_b = run_set_b_item(
        DeterministicECClient(), CHIPMUNKS, 55.0, samples_per_prompt=32, temperature=1.0
    )
    clusterer = InferenceClusterer(DeterministicStructuredClient())
    first = run_clustering_item(clusterer, source_a, _provenance(), pilot=True)
    second = json.loads(json.dumps(first))
    second["id"] = "second-A"
    second["realized_gain"]["full"] = 0.25
    second["realized_gain"]["split_first"] = 0.25
    second["realized_gain"]["split_second"] = 0.25
    for arm in ("oracle", "q", "qs", "answer_string"):
        second["estimates"][arm]["full"]["between_reading_variance"] /= 2
        second["estimates"][arm]["split_first"]["between_reading_variance"] /= 2
        second["estimates"][arm]["split_second"]["between_reading_variance"] /= 2
    control = run_clustering_item(clusterer, source_b, _provenance(), pilot=True)
    summary = analyze_clustering(
        [first, second, control], bootstrap_samples=100, bootstrap_seed=7
    )
    assert summary["pilot"] is True
    assert summary["analysis_status"] == "pilot_only_not_confirmatory"
    assert set(summary["split_half_correlations"]) == {"oracle", "q", "qs", "answer_string"}
    assert set(summary["split_half_correlations"]["q"]) == {
        "same_samples",
        "split_first_half_predictor",
        "split_second_half_predictor",
    }
    assert summary["lister_cluster_count_distributions"]["q"]["setB"]["1"] == 1
    assert summary["setA_agreement_audit"]["q"]["item_recall_rate"] == 1.0
    assert set(summary["setB_between_batch_diagnostic"]["mean_by_arm"]) == {
        "oracle",
        "q",
        "qs",
        "answer_string",
    }
