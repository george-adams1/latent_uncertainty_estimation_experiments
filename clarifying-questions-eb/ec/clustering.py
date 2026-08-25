"""Annotation-blind interpretation clustering over saved E-C samples.

The LLM-facing functions in this module accept only the original question,
the ambiguous/original samples (for QS), inferred interpretations, and one
answer at a time. Oracle labels are used only after inference for auditing and
for the unchanged E-C outcome calculation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from eb.grading import normalize_text
from ec.estimator import categorical_variance, decompose_conditionals, empirical_distribution
from ec.split_half import HALF, _gain


SCHEMA_VERSION = "ec.clustering.v2"
PROMPT_DIR = Path(__file__).with_name("prompts")
PROMPT_FILES = {
    "lister_system": "clustering_lister_system.txt",
    "lister_q_user": "clustering_lister_q_user.txt",
    "lister_qs_user": "clustering_lister_qs_user.txt",
    "assigner_system": "clustering_assigner_system.txt",
    "assigner_user": "clustering_assigner_user.txt",
}


class StructuredCompletionClient(Protocol):
    def complete_json(
        self,
        system: str,
        user: str,
        schema_name: str,
        schema: dict,
        max_tokens: int,
    ) -> tuple[dict, str]: ...


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def load_prompts(prompt_dir: str | Path = PROMPT_DIR) -> tuple[dict[str, str], dict[str, str]]:
    directory = Path(prompt_dir)
    prompts = {name: (directory / filename).read_text() for name, filename in PROMPT_FILES.items()}
    hashes = {name: _sha256_bytes(text.encode()) for name, text in prompts.items()}
    return prompts, hashes


def _render(template: str, **values: str) -> str:
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{{" + name + "}}", value)
    leftovers = re.findall(r"\{\{[A-Z_]+\}\}", rendered)
    if leftovers:
        raise ValueError(f"unfilled prompt placeholders: {leftovers}")
    return rendered


def lister_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "interpretations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"type": "string", "minLength": 1},
            }
        },
        "required": ["interpretations"],
        "additionalProperties": False,
    }


def assignment_schema(valid_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"assignment": {"type": "string", "enum": valid_ids + ["none"]}},
        "required": ["assignment"],
        "additionalProperties": False,
    }


def _validate_interpretations(payload: dict) -> list[dict[str, str]]:
    texts = payload.get("interpretations")
    if not isinstance(texts, list) or not 1 <= len(texts) <= 4:
        raise ValueError("lister must return between one and four interpretations")
    cleaned: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each interpretation must be a nonempty string")
        value = " ".join(text.split())
        key = value.casefold()
        if key in seen:
            raise ValueError("lister returned duplicate interpretations")
        seen.add(key)
        cleaned.append(value)
    return [{"id": f"I{index}", "text": text} for index, text in enumerate(cleaned, 1)]


class InferenceClusterer:
    def __init__(self, client: StructuredCompletionClient, prompt_dir: str | Path = PROMPT_DIR):
        self.client = client
        self.prompts, self.prompt_hashes = load_prompts(prompt_dir)

    def list_interpretations(
        self, question: str, samples: list[str] | None
    ) -> dict:
        variant = "q" if samples is None else "qs"
        template = self.prompts[f"lister_{variant}_user"]
        values = {"QUESTION_JSON": json.dumps(question, ensure_ascii=False)}
        if samples is not None:
            values["SAMPLES_JSON"] = json.dumps(samples, ensure_ascii=False)
        user = _render(template, **values)
        payload, raw = self.client.complete_json(
            self.prompts["lister_system"], user, f"ec_lister_{variant}", lister_schema(), 384
        )
        return {"interpretations": _validate_interpretations(payload), "raw_response": raw}

    def assign(self, question: str, interpretations: list[dict[str, str]], answer: str) -> dict:
        valid_ids = [item["id"] for item in interpretations]
        user = _render(
            self.prompts["assigner_user"],
            QUESTION_JSON=json.dumps(question, ensure_ascii=False),
            INTERPRETATIONS_JSON=json.dumps(interpretations, ensure_ascii=False),
            ANSWER_JSON=json.dumps(answer, ensure_ascii=False),
        )
        payload, raw = self.client.complete_json(
            self.prompts["assigner_system"],
            user,
            "ec_assignment",
            assignment_schema(valid_ids),
            32,
        )
        label = payload.get("assignment")
        if label not in valid_ids + ["none"]:
            raise ValueError(f"invalid assignment {label!r}; expected one of {valid_ids + ['none']}")
        return {"label": label, "raw_response": raw}


def _record_layout(record: dict) -> tuple[str, dict[str, list[str]], dict[str, str]]:
    """Return original question, saved sample batches, and E-C condition mapping."""
    if record["set"] == "A":
        batches = {"ambiguous": record["ambiguous_prompt"]["samples"]}
        conditions = {}
        for name, reading in record["readings"].items():
            batch_name = f"condition:{name}"
            batches[batch_name] = reading["prompt"]["samples"]
            conditions[name] = batch_name
        return record["ambiguous_question"], batches, conditions
    batches = {
        "original": record["original_prompt"]["samples"],
        "condition:reading": record["repeat_prompt"]["samples"],
    }
    return record["question"], batches, {"reading": "condition:reading"}


def _oracle_labels(record: dict) -> dict[str, list[str]]:
    if record["set"] == "A":
        labels = {"ambiguous": record["ambiguous_prompt"]["clusters"]}
        labels.update(
            {
                f"condition:{name}": reading["prompt"]["clusters"]
                for name, reading in record["readings"].items()
            }
        )
        return labels
    return {
        "original": record["original_prompt"]["clusters"],
        "condition:reading": record["repeat_prompt"]["clusters"],
    }


def _answer_string_labels(batches: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
    normalized = sorted({normalize_text(answer) for answers in batches.values() for answer in answers})
    key_by_text = {text: f"S{index:03d}" for index, text in enumerate(normalized, 1)}
    labels = {
        batch: [key_by_text[normalize_text(answer)] for answer in answers]
        for batch, answers in batches.items()
    }
    return labels, {label: text for text, label in key_by_text.items()}


def cluster_sizes(labels: list[str], categories: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {category: counts.get(category, 0) for category in categories}


def estimate_from_labels(
    labels_by_batch: dict[str, list[str]],
    condition_batches: dict[str, str],
    categories: list[str],
    lo: int | None = None,
    hi: int | None = None,
) -> dict:
    conditionals = {}
    for condition, batch in condition_batches.items():
        labels = labels_by_batch[batch][slice(lo, hi)]
        conditionals[condition] = empirical_distribution(labels, categories)
    return decompose_conditionals(conditionals)


def observed_variance(labels: list[str], categories: list[str]) -> float:
    return categorical_variance(empirical_distribution(labels, categories))


def _arm_estimates(
    labels: dict[str, list[str]],
    categories: list[str],
    observed_batch: str,
    conditions: dict[str, str],
) -> dict:
    lengths = {len(labels[batch]) for batch in conditions.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2 * HALF:
        raise ValueError("every condition batch must contain at least 32 aligned samples")
    return {
        "full": estimate_from_labels(labels, conditions, categories),
        "split_first": estimate_from_labels(labels, conditions, categories, 0, HALF),
        "split_second": estimate_from_labels(labels, conditions, categories, HALF, 2 * HALF),
        "observed_categorical_variance": observed_variance(labels[observed_batch], categories),
    }


def _best_oracle_alignment(
    inferred: dict[str, list[str]],
    oracle: dict[str, list[str]],
    interpretation_ids: list[str],
    reading_names: list[str],
) -> dict:
    """Align arbitrary inferred IDs to readings by maximum sample overlap.

    This is an annotation-using audit only. It is never fed back into the
    estimator or any clusterer prompt.
    """
    pairs = [
        (inferred[batch][index], oracle[batch][index])
        for batch in inferred
        for index in range(len(inferred[batch]))
    ]
    best_score = -1
    best_mapping: dict[str, str] = {}
    for count in range(min(len(interpretation_ids), len(reading_names)) + 1):
        for ids in itertools.combinations(interpretation_ids, count):
            for readings in itertools.permutations(reading_names, count):
                mapping = dict(zip(ids, readings))
                score = sum(mapping.get(inferred_label) == oracle_label for inferred_label, oracle_label in pairs)
                if score > best_score:
                    best_score = score
                    best_mapping = mapping
    match_count = sum(
        ("other" if inferred_label == "none" else best_mapping.get(inferred_label)) == oracle_label
        for inferred_label, oracle_label in pairs
    )
    unique_pairs = [(a, b) for a, b in pairs if b in reading_names]
    unique_matches = sum(best_mapping.get(a) == b for a, b in unique_pairs)
    recovered = sorted(set(best_mapping.values()))
    return {
        "method": "maximum one-to-one sample-overlap alignment; none is compared with oracle other",
        "mapping": best_mapping,
        "matches_all": match_count,
        "n_all": len(pairs),
        "agreement_all": match_count / len(pairs) if pairs else None,
        "matches_unique_oracle": unique_matches,
        "n_unique_oracle": len(unique_pairs),
        "agreement_unique_oracle": unique_matches / len(unique_pairs) if unique_pairs else None,
        "oracle_readings_recovered": recovered,
        "oracle_reading_recall": len(recovered) / len(reading_names) if reading_names else None,
    }


def run_clustering_item(
    clusterer: InferenceClusterer,
    source: dict,
    provenance: dict,
    pilot: bool,
) -> dict:
    question, batches, conditions = _record_layout(source)
    observed_batch = "ambiguous" if source["set"] == "A" else "original"
    listers = {
        "q": clusterer.list_interpretations(question, None),
        "qs": clusterer.list_interpretations(question, batches[observed_batch]),
    }

    arm_assignments: dict[str, dict[str, list[dict]]] = {}
    labels_by_arm: dict[str, dict[str, list[str]]] = {}
    categories_by_arm: dict[str, list[str]] = {}
    for variant, lister in listers.items():
        interpretations = lister["interpretations"]
        categories = [item["id"] for item in interpretations] + ["none"]
        assigned = {
            batch: [clusterer.assign(question, interpretations, answer) for answer in answers]
            for batch, answers in batches.items()
        }
        arm_assignments[variant] = assigned
        labels_by_arm[variant] = {
            batch: [assignment["label"] for assignment in assignments]
            for batch, assignments in assigned.items()
        }
        categories_by_arm[variant] = categories

    oracle = _oracle_labels(source)
    if source["set"] == "A":
        oracle_categories = list(source["readings"]) + ["multiple", "other"]
    else:
        oracle_categories = ["reading", "multiple", "other"]
    string_labels, string_map = _answer_string_labels(batches)
    string_categories = list(string_map)
    labels_by_arm.update({"oracle": oracle, "answer_string": string_labels})
    categories_by_arm.update({"oracle": oracle_categories, "answer_string": string_categories})

    estimates = {
        arm: _arm_estimates(labels, categories_by_arm[arm], observed_batch, conditions)
        for arm, labels in labels_by_arm.items()
    }
    set_b_batch_diagnostic = None
    if source["set"] == "B":
        batch_conditions = {
            "original_batch": "original",
            "repeat_batch": "condition:reading",
        }
        set_b_batch_diagnostic = {
            "name": "between_batch_variance",
            "interpretation": (
                "finite-sample separation between two independent batches from the same "
                "unambiguous question; this is not between-reading variance"
            ),
            "arms": {
                arm: _arm_estimates(
                    labels, categories_by_arm[arm], observed_batch, batch_conditions
                )
                for arm, labels in labels_by_arm.items()
            },
        }
    oracle_delta = (
        estimates["oracle"]["full"]["between_reading_variance"]
        - source["estimator"]["between_reading_variance"]
    )
    if abs(oracle_delta) > 1e-12:
        raise ValueError(f"refactored oracle estimator changed the source result by {oracle_delta}")

    cluster_size_records = {
        arm: {
            batch: cluster_sizes(labels, categories_by_arm[arm])
            for batch, labels in labels_by_batch.items()
        }
        for arm, labels_by_batch in labels_by_arm.items()
    }
    set_a_audit = None
    if source["set"] == "A":
        reading_names = list(source["readings"])
        set_a_audit = {}
        for variant in ("q", "qs"):
            ids = [item["id"] for item in listers[variant]["interpretations"]]
            alignment = _best_oracle_alignment(labels_by_arm[variant], oracle, ids, reading_names)
            alignment["none_count"] = sum(label == "none" for labels in labels_by_arm[variant].values() for label in labels)
            set_a_audit[variant] = alignment
        flattened_oracle = [label for labels in oracle.values() for label in labels]
        set_a_audit["oracle_special_counts"] = {
            "multiple": flattened_oracle.count("multiple"),
            "other": flattened_oracle.count("other"),
            "n": len(flattened_oracle),
        }

    gains = {"full": source["realized_gain"]}
    if source["set"] == "A":
        gains.update({
            "split_first": _gain(source, 0, HALF),
            "split_second": _gain(source, HALF, 2 * HALF),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "pilot": pilot,
        "analysis_status": "pilot_only_not_confirmatory" if pilot else "frozen_full_run",
        "set": source["set"],
        "id": source["id"],
        "question": question,
        "subject_model": provenance["subject_model"],
        "clusterer": provenance["clusterer"],
        "source": provenance["source"],
        "listers": listers,
        "assignments": arm_assignments,
        "oracle_assignments": oracle,
        "answer_string_assignments": string_labels,
        "answer_string_cluster_text": string_map,
        "cluster_sizes": cluster_size_records,
        "estimates": estimates,
        "set_b_batch_diagnostic": set_b_batch_diagnostic,
        "realized_gain": gains,
        "set_a_agreement_audit": set_a_audit,
    }
