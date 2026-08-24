import copy
import io
import json
import urllib.error

import pytest

from eb.model_client import ModelClient
from ec.analyze import analyze
from ec.estimator import categorical_variance, decompose_conditionals, empirical_distribution
from ec.run_experiment import OpenAICompatibleClient, run_set_a_item, run_set_b_item

from .fixtures import CHIPMUNKS, GEORGIA


class DeterministicECClient(ModelClient):
    """Typed fixture with a genuinely mixed ambiguous answer distribution."""

    def __init__(self):
        self.ambiguous_index = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        if "country Georgia" in user:
            return "Tbilisi"
        if "U.S. state Georgia" in user:
            return "Atlanta"
        if "capital of Georgia" in user:
            answers = ["Tbilisi", "Atlanta"]
            answer = answers[self.ambiguous_index % len(answers)]
            self.ambiguous_index += 1
            return answer
        if "Chipmunks" in user:
            return "David Seville"
        return "unknown"


def test_categorical_variance_decomposition_identity():
    conditionals = {
        "reading_a": {"reading_a": 1.0, "reading_b": 0.0},
        "reading_b": {"reading_a": 0.0, "reading_b": 1.0},
    }
    result = decompose_conditionals(conditionals)
    assert result["total_variance"] == pytest.approx(0.5)
    assert result["within_reading_variance"] == pytest.approx(0.0)
    assert result["between_reading_variance"] == pytest.approx(0.5)
    assert result["between_fraction"] == pytest.approx(1.0)
    assert result["identity_error"] == pytest.approx(0.0)


def test_identical_conditionals_have_no_between_reading_variance():
    conditionals = {
        "reading_a": {"reading_a": 0.75, "other": 0.25},
        "reading_b": {"reading_a": 0.75, "other": 0.25},
    }
    result = decompose_conditionals(conditionals)
    assert result["within_reading_variance"] == pytest.approx(result["total_variance"])
    assert result["between_reading_variance"] == pytest.approx(0.0)


def test_empirical_distribution_retains_zero_categories():
    distribution = empirical_distribution(["a", "a", "b"], ["a", "b", "other"])
    assert distribution == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3), "other": 0.0}
    assert categorical_variance(distribution) == pytest.approx(4 / 9)


def test_set_a_ec_record_estimates_between_reading_component_and_gain():
    record = run_set_a_item(
        DeterministicECClient(),
        GEORGIA,
        confidence=55.0,
        samples_per_prompt=4,
        temperature=1.0,
    )
    assert record["schema_version"] == "ec.v1"
    assert record["ambiguous_prompt"]["samples"] == ["Tbilisi", "Atlanta", "Tbilisi", "Atlanta"]
    assert record["estimator"]["observed_ambiguous_variance"] == pytest.approx(0.5)
    assert record["estimator"]["within_reading_variance"] == pytest.approx(0.0)
    assert record["estimator"]["between_reading_variance"] == pytest.approx(0.5)
    assert record["readings"]["reading_a"]["baseline_accuracy"] == pytest.approx(0.5)
    assert record["readings"]["reading_a"]["clarified_accuracy"] == pytest.approx(1.0)
    assert record["readings"]["reading_b"]["baseline_accuracy"] == pytest.approx(0.5)
    assert record["readings"]["reading_b"]["clarified_accuracy"] == pytest.approx(1.0)
    assert record["realized_gain"] == pytest.approx(0.5)
    # Full prompt inputs and raw samples are retained for audit/reanalysis.
    assert record["readings"]["reading_a"]["question"] == GEORGIA.reading_a.question
    assert len(record["readings"]["reading_b"]["prompt"]["samples"]) == 4
    json.dumps(record)


def test_set_b_repeat_control_has_zero_between_component_and_gain():
    record = run_set_b_item(
        DeterministicECClient(),
        CHIPMUNKS,
        confidence=55.0,
        samples_per_prompt=4,
        temperature=1.0,
    )
    assert record["estimator"]["between_reading_variance"] == pytest.approx(0.0)
    assert record["original_accuracy"] == pytest.approx(1.0)
    assert record["repeat_accuracy"] == pytest.approx(1.0)
    assert record["realized_gain"] == pytest.approx(0.0)
    assert len(record["original_prompt"]["samples"]) == 4
    assert len(record["repeat_prompt"]["samples"]) == 4


def test_openai_compatible_client_retries_transient_failure(monkeypatch):
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary")
        return io.BytesIO(b'{"choices":[{"message":{"content":" Atlanta "}}]}')

    monkeypatch.setattr("ec.run_experiment.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("ec.run_experiment.time.sleep", lambda seconds: None)
    client = OpenAICompatibleClient("http://localhost:8000/v1", "fixture", retries=1)
    assert client.complete("system", "user", temperature=1.0) == "Atlanta"
    assert calls == 2


def test_ec_analysis_reports_primary_set_contrasts_deterministically():
    set_a = run_set_a_item(
        DeterministicECClient(), GEORGIA, 55.0, samples_per_prompt=4, temperature=1.0
    )
    set_b = run_set_b_item(
        DeterministicECClient(), CHIPMUNKS, 55.0, samples_per_prompt=4, temperature=1.0
    )
    # A second item in each stratum prevents every resampled correlation from
    # being undefined while keeping the primary contrast transparent.
    set_a_2 = copy.deepcopy(set_a)
    set_a_2["id"] = "qA2"
    set_a_2["confidence"] = 60.0
    set_a_2["realized_gain"] = 0.25
    set_a_2["estimator"]["between_reading_variance"] = 0.25
    set_b_2 = copy.deepcopy(set_b)
    set_b_2["id"] = "qB2"
    set_b_2["confidence"] = 60.0

    first = analyze([set_a, set_a_2, set_b, set_b_2], bootstrap_samples=200, bootstrap_seed=9)
    second = analyze([set_a, set_a_2, set_b, set_b_2], bootstrap_samples=200, bootstrap_seed=9)
    assert first["means"]["setA"]["between_reading_variance"] == pytest.approx(0.375)
    assert first["contrasts"]["setA_minus_setB_between_reading_variance"] == pytest.approx(0.375)
    assert first["contrasts"]["setA_minus_setB_realized_gain"] == pytest.approx(0.375)
    assert "setA_between_minus_observed_variance_gain_correlation" in first["correlation_contrasts"]
    assert first["max_abs_decomposition_identity_error"] == pytest.approx(0.0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
