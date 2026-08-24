"""Run E-C from a completed E-B matched sample.

E-C repeatedly samples the original question and every fixed-reading rewrite,
retains the raw responses, decomposes categorical answer variability into
within- and between-reading components, and measures the realized accuracy
gain from fixing the reading. Set B receives two independent batches of the
same question as a null control.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eb.conditions import ANSWER_SYSTEM
from eb.data_ambigqa import load_ambigqa_candidates
from eb.grading import contains_alias, grade_ambiguous
from eb.model_client import LocalHFClient, ModelClient
from ec.analyze import analyze
from ec.estimator import (
    categorical_variance,
    decompose_conditionals,
    empirical_distribution,
    total_variation,
)


SCHEMA_VERSION = "ec.v1"


class OpenAICompatibleClient(ModelClient):
    """Minimal stateless chat client for vLLM or another compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 200,
        timeout: float = 120.0,
        retries: int = 3,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                return (payload["choices"][0]["message"].get("content") or "").strip()
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"completion request failed after {self.retries + 1} attempts: {error}"
                    ) from error
                time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


def sample_prompt(
    client: ModelClient,
    question: str,
    n: int,
    temperature: float,
) -> dict:
    if n <= 0:
        raise ValueError("sample count must be positive")
    user = f"Question: {question}"
    samples = [client.complete(ANSWER_SYSTEM, user, temperature=temperature) for _ in range(n)]
    return {
        "system_prompt": ANSWER_SYSTEM,
        "user_prompt": user,
        "temperature": temperature,
        "samples": samples,
    }


def cluster_response(response: str, readings: dict[str, list[str]]) -> str:
    hits = [name for name, aliases in readings.items() if contains_alias(response, aliases)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "multiple"
    return "other"


def add_clusters(prompt_record: dict, readings: dict[str, list[str]]) -> dict:
    categories = list(readings) + ["multiple", "other"]
    clusters = [cluster_response(sample, readings) for sample in prompt_record["samples"]]
    distribution = empirical_distribution(clusters, categories)
    return {
        **prompt_record,
        "clusters": clusters,
        "cluster_distribution": distribution,
        "categorical_variance": categorical_variance(distribution),
    }


def strict_accuracy(samples: list[str], intended: str, readings: dict[str, list[str]]) -> float:
    intended_aliases = readings[intended]
    other_aliases = [
        alias
        for name, aliases in readings.items()
        if name != intended
        for alias in aliases
    ]
    return sum(
        grade_ambiguous(sample, intended_aliases, other_aliases) == "correct"
        for sample in samples
    ) / len(samples)


def run_set_a_item(
    client: ModelClient,
    item,
    confidence: float,
    samples_per_prompt: int,
    temperature: float,
    source_eb_record: dict | None = None,
) -> dict:
    readings = {
        "reading_a": item.reading_a.answers,
        "reading_b": item.reading_b.answers,
    }
    ambiguous = add_clusters(
        sample_prompt(client, item.ambiguous_question, samples_per_prompt, temperature),
        readings,
    )
    reading_data = {}
    conditional_distributions = {}
    for name, reading in (("reading_a", item.reading_a), ("reading_b", item.reading_b)):
        prompted = add_clusters(
            sample_prompt(client, reading.question, samples_per_prompt, temperature),
            readings,
        )
        baseline_accuracy = strict_accuracy(ambiguous["samples"], name, readings)
        clarified_accuracy = strict_accuracy(prompted["samples"], name, readings)
        reading_data[name] = {
            "question": reading.question,
            "answer_aliases": reading.answers,
            "prompt": prompted,
            "baseline_accuracy": baseline_accuracy,
            "clarified_accuracy": clarified_accuracy,
            "realized_gain": clarified_accuracy - baseline_accuracy,
        }
        conditional_distributions[name] = prompted["cluster_distribution"]

    estimator = decompose_conditionals(conditional_distributions)
    estimator["observed_ambiguous_distribution"] = ambiguous["cluster_distribution"]
    estimator["observed_ambiguous_variance"] = ambiguous["categorical_variance"]
    estimator["mixture_fit_total_variation"] = total_variation(
        ambiguous["cluster_distribution"],
        estimator["mixture_distribution"],
    )
    realized_gain = sum(data["realized_gain"] for data in reading_data.values()) / len(reading_data)
    return {
        "schema_version": SCHEMA_VERSION,
        "set": "A",
        "id": item.id,
        "confidence": confidence,
        "ambiguous_question": item.ambiguous_question,
        "ambiguous_prompt": ambiguous,
        "readings": reading_data,
        "estimator": estimator,
        "realized_gain": realized_gain,
        "source_eb": {
            "intended": source_eb_record.get("intended") if source_eb_record else None,
            "self_ask_leaked": bool(source_eb_record and source_eb_record["self_ask"].get("leaked")),
        },
    }


def run_set_b_item(
    client: ModelClient,
    item,
    confidence: float,
    samples_per_prompt: int,
    temperature: float,
) -> dict:
    readings = {"reading": item.answers}
    original = add_clusters(
        sample_prompt(client, item.question, samples_per_prompt, temperature),
        readings,
    )
    repeat = add_clusters(
        sample_prompt(client, item.question, samples_per_prompt, temperature),
        readings,
    )
    original_accuracy = sum(contains_alias(sample, item.answers) for sample in original["samples"]) / len(original["samples"])
    repeat_accuracy = sum(contains_alias(sample, item.answers) for sample in repeat["samples"]) / len(repeat["samples"])
    estimator = decompose_conditionals({"reading": repeat["cluster_distribution"]})
    estimator["observed_ambiguous_distribution"] = original["cluster_distribution"]
    estimator["observed_ambiguous_variance"] = original["categorical_variance"]
    estimator["mixture_fit_total_variation"] = total_variation(
        original["cluster_distribution"],
        estimator["mixture_distribution"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "set": "B",
        "id": item.id,
        "confidence": confidence,
        "question": item.question,
        "answer_aliases": item.answers,
        "original_prompt": original,
        "repeat_prompt": repeat,
        "original_accuracy": original_accuracy,
        "repeat_accuracy": repeat_accuracy,
        "estimator": estimator,
        "realized_gain": repeat_accuracy - original_accuracy,
    }


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_items(eb_results_path: str, set_b_screen_path: str, include_leaked: bool = False):
    eb_records = _load_jsonl(eb_results_path)
    selected_a = [row for row in eb_records if row["set"] == "A"]
    if not include_leaked:
        selected_a = [row for row in selected_a if not row["self_ask"].get("leaked", False)]
    selected_b = [row for row in eb_records if row["set"] == "B"]

    ambig_pool = load_ambigqa_candidates("validation") + load_ambigqa_candidates("train")
    ambig_by_id = {item.id: item for item in ambig_pool}
    missing_a = [row["id"] for row in selected_a if row["id"] not in ambig_by_id]
    if missing_a:
        raise ValueError(f"could not resolve Set A IDs: {missing_a}")

    screen_rows = _load_jsonl(set_b_screen_path)
    screen_by_id = {row["id"]: row for row in screen_rows}
    missing_b = [row["id"] for row in selected_b if row["id"] not in screen_by_id]
    if missing_b:
        raise ValueError(f"could not resolve Set B IDs: {missing_b}")

    from eb.data_setb import SetBItem

    set_a = [(ambig_by_id[row["id"]], float(row["confidence"]), row) for row in selected_a]
    set_b = [
        (
            SetBItem(
                id=row["id"],
                question=screen_by_id[row["id"]]["question"],
                answers=screen_by_id[row["id"]]["answers"],
            ),
            float(row["confidence"]),
        )
        for row in selected_b
    ]
    return set_a, set_b


def _build_client(args) -> ModelClient:
    if args.mock:
        from tests.fixtures import make_mock_client

        return make_mock_client(args.mock)
    if args.base_url:
        return OpenAICompatibleClient(
            args.base_url,
            args.model,
            args.max_tokens,
            args.timeout,
            args.retries,
        )
    return LocalHFClient(args.model, max_new_tokens=args.max_tokens, device_map=args.device_map)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb-results", required=True)
    parser.add_argument("--setb-screen", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--base-url", help="OpenAI-compatible /v1 base URL, e.g. a vLLM server")
    parser.add_argument("--mock", choices=["typed", "flat"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1, help="parallel items; use >1 only with a concurrent API server")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit-per-set", type=int, default=None)
    parser.add_argument("--include-leaked", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args(argv)
    if args.workers > 1 and not args.base_url:
        parser.error("--workers > 1 is supported only with --base-url")
    if args.retries < 0:
        parser.error("--retries must be nonnegative")

    set_a, set_b = resolve_items(args.eb_results, args.setb_screen, args.include_leaked)
    if args.limit_per_set is not None:
        set_a = set_a[: args.limit_per_set]
        set_b = set_b[: args.limit_per_set]
    client = _build_client(args)

    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (item, confidence, source) in enumerate(set_a):
            future = executor.submit(
                run_set_a_item,
                client,
                item,
                confidence,
                args.samples_per_prompt,
                args.temperature,
                source,
            )
            tasks.append((index, future))
        offset = len(set_a)
        for index, (item, confidence) in enumerate(set_b):
            future = executor.submit(
                run_set_b_item,
                client,
                item,
                confidence,
                args.samples_per_prompt,
                args.temperature,
            )
            tasks.append((offset + index, future))
        records: list[dict | None] = [None] * len(tasks)
        future_to_index = {future: index for index, future in tasks}
        for completed, future in enumerate(as_completed(future_to_index), 1):
            records[future_to_index[future]] = future.result()
            if completed % 10 == 0 or completed == len(tasks):
                print(f"E-C items completed={completed}/{len(tasks)}", flush=True)

    final_records = [row for row in records if row is not None]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in final_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = analyze(
        final_records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "samples_per_prompt": args.samples_per_prompt,
            "temperature": args.temperature,
            "source_eb_results": args.eb_results,
            "source_setb_screen": args.setb_screen,
            "excluded_leaked_setA": not args.include_leaked,
            "results_path": str(output),
        }
    )
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
