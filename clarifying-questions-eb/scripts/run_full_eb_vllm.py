#!/usr/bin/env python3
"""Run the complete E-B experiment from a saved full Set A screening scan."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eb.analyze import analyze, bootstrap_ci, wilson_ci
from eb.data_ambigqa import load_ambigqa_candidates
from eb.data_setb import load_setb_candidates
from eb.run_experiment import run_set_a_item, run_set_b_item
from eb.screening import MatchedSetAItem, MatchedSetBItem


ANSWER_SYSTEM = "Answer the question as concisely as possible: a short phrase, no explanation."
CONFIDENCE_SYSTEM = (
    "You will be shown a question and a proposed answer to it. State only your "
    "confidence that the proposed answer is correct, as a single integer "
    "percentage from 0 to 100. Respond with only the number, nothing else."
)


def parse_confidence(response: str) -> float | None:
    match = re.search(r"\d{1,3}", response)
    if not match:
        return None
    value = float(match.group())
    return value if 0 <= value <= 100 else None


class VLLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str, temperature: float = 0.0, retries: int = 5) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                return (payload["choices"][0]["message"].get("content") or "").strip()
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                if attempt + 1 == retries:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def load_matched_set_a(scan_path: Path, rng: random.Random) -> list[MatchedSetAItem]:
    pool = load_ambigqa_candidates("validation") + load_ambigqa_candidates("train")
    rows = [json.loads(line) for line in scan_path.open() if line.strip()]
    if len(rows) != len(pool) or len(pool) != 2956:
        raise RuntimeError(f"Set A scan/pool mismatch: {len(rows)} rows, {len(pool)} candidates")
    result = []
    for index, (row, item) in enumerate(zip(rows, pool)):
        if row["index"] != index or row["id"] != item.id or row["question"] != item.ambiguous_question:
            raise RuntimeError(f"Set A alignment mismatch at index {index}")
        if not row["matched"]:
            continue
        result.append(
            MatchedSetAItem(
                id=item.id,
                ambiguous_question=item.ambiguous_question,
                reading_a=item.reading_a,
                reading_b=item.reading_b,
                intended=rng.choice(["a", "b"]),
                confidence=row["confidence"],
                answer_now_response=row["answer"],
            )
        )
    return result


def screen_set_b_one(client: VLLMClient, split: str, index: int, item, low: float, high: float) -> dict:
    answer = client.complete(ANSWER_SYSTEM, f"Question: {item.question}", temperature=0.0)
    raw_confidence = client.complete(
        CONFIDENCE_SYSTEM,
        f"Question: {item.question}\nProposed answer: {answer}",
        temperature=0.0,
    )
    confidence = parse_confidence(raw_confidence)
    return {
        "split": split,
        "index": index,
        "id": item.id,
        "question": item.question,
        "answers": item.answers,
        "answer": answer,
        "confidence_response": raw_confidence,
        "confidence": confidence,
        "matched": confidence is not None and low <= confidence <= high,
    }


def screen_set_b(
    client: VLLMClient,
    target: int,
    workers: int,
    low: float,
    high: float,
    checkpoint_path: Path,
) -> tuple[list[MatchedSetBItem], list[dict]]:
    all_rows: list[dict] = []
    matched_rows: list[dict] = []
    for split in ("validation", "train"):
        pool = load_setb_candidates(split)
        for start in range(0, len(pool), 4096):
            chunk = pool[start : start + 4096]
            rows: list[dict | None] = [None] * len(chunk)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(screen_set_b_one, client, split, start + offset, item, low, high): offset
                    for offset, item in enumerate(chunk)
                }
                for future in as_completed(futures):
                    row = future.result()
                    rows[row["index"] - start] = row
            completed_rows = [row for row in rows if row is not None]
            all_rows.extend(completed_rows)
            matched_rows.extend(row for row in completed_rows if row["matched"])
            with checkpoint_path.open("w") as handle:
                for row in all_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"Set B screened={len(all_rows)} matched={len(matched_rows)} target={target}",
                flush=True,
            )
            if len(matched_rows) >= target:
                selected = matched_rows[:target]
                return [
                    MatchedSetBItem(
                        id=row["id"],
                        question=row["question"],
                        answers=row["answers"],
                        confidence=row["confidence"],
                        answer_now_response=row["answer"],
                    )
                    for row in selected
                ], all_rows
    selected = matched_rows[:target]
    return [
        MatchedSetBItem(
            id=row["id"],
            question=row["question"],
            answers=row["answers"],
            confidence=row["confidence"],
            answer_now_response=row["answer"],
        )
        for row in selected
    ], all_rows


def run_items(client: VLLMClient, set_a, set_b, workers: int, diagnostic_n: int) -> list[dict]:
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, item in enumerate(set_a):
            tasks.append((index, executor.submit(run_set_a_item, client, item, diagnostic_n, 1.0)))
        offset = len(set_a)
        for index, item in enumerate(set_b):
            tasks.append((offset + index, executor.submit(run_set_b_item, client, item, diagnostic_n, 1.0)))
        results: list[dict | None] = [None] * len(tasks)
        for completed, (index, future) in enumerate(tasks, 1):
            results[index] = future.result()
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Experiment items completed={completed}/{len(tasks)}", flush=True)
    return [record for record in results if record is not None]


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator


def extended_summary(
    model: str,
    records: list[dict],
    set_b_screen_rows: list[dict],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict:
    base = analyze(records, bootstrap_samples, bootstrap_seed, confidence_level)
    set_a = [row for row in records if row["set"] == "A"]
    set_b = [row for row in records if row["set"] == "B"]
    pooled = set_a + set_b
    strict = {"correct": 1.0, "wrong": 0.0, "hedged": 0.0}
    gains = [strict[row["self_ask"]["grade"]] - strict[row["answer_now"]["grade"]] for row in pooled]
    asked = [float(row["free_choice"]["asked"]) for row in pooled]
    def condition_accuracy(rows: list[dict], condition: str) -> float:
        return sum(strict[row[condition]["grade"]] for row in rows) / len(rows)

    def condition_gain(rows: list[dict], condition: str) -> float:
        return sum(
            strict[row[condition]["grade"]] - strict[row["answer_now"]["grade"]]
            for row in rows
        ) / len(rows)

    free_choice = {
        "setA_accuracy": condition_accuracy(set_a, "free_choice"),
        "setB_accuracy": condition_accuracy(set_b, "free_choice"),
        "setA_gain": condition_gain(set_a, "free_choice"),
        "setB_gain": condition_gain(set_b, "free_choice"),
        "setA_ask_rate": sum(row["free_choice"]["asked"] for row in set_a) / len(set_a),
        "setB_ask_rate": sum(row["free_choice"]["asked"] for row in set_b) / len(set_b),
        "confidence_vs_asked_corr": corr([row["confidence"] for row in pooled], asked),
        "realized_gain_vs_asked_corr": corr(gains, asked),
    }
    base.update(
        {
            "model": model,
            "setB_screened_count": len(set_b_screen_rows),
            "setB_screened_match_count": sum(row["matched"] for row in set_b_screen_rows),
            "self_ask_leaks_setA": sum(row["self_ask"]["leaked"] for row in set_a),
            "free_choice_leaks_setA": sum(bool(row["free_choice"]["leaked"]) for row in set_a),
            "free_choice": free_choice,
        }
    )
    ci = base["confidence_intervals"]
    ci["free_choice"] = {}
    for set_name, rows in (("setA", set_a), ("setB", set_b)):
        ci["free_choice"][f"{set_name}_accuracy"] = wilson_ci(
            sum(row["free_choice"]["grade"] == "correct" for row in rows),
            len(rows),
            confidence_level,
        )
        ci["free_choice"][f"{set_name}_gain"] = bootstrap_ci(
            [rows],
            lambda sampled: condition_gain(sampled, "free_choice"),
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            label=f"free_choice:{set_name}_gain",
        )
        ci["free_choice"][f"{set_name}_ask_rate"] = wilson_ci(
            sum(row["free_choice"]["asked"] for row in rows),
            len(rows),
            confidence_level,
        )
    ci["free_choice"]["confidence_vs_asked_corr"] = bootstrap_ci(
        [set_a, set_b],
        lambda a, b: corr(
            [row["confidence"] for row in a + b],
            [float(row["free_choice"]["asked"]) for row in a + b],
        ),
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        label="free_choice:confidence_vs_asked_corr",
    )
    ci["free_choice"]["realized_gain_vs_asked_corr"] = bootstrap_ci(
        [set_a, set_b],
        lambda a, b: corr(
            [
                strict[row["self_ask"]["grade"]] - strict[row["answer_now"]["grade"]]
                for row in a + b
            ],
            [float(row["free_choice"]["asked"]) for row in a + b],
        ),
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        label="free_choice:realized_gain_vs_asked_corr",
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seta-scan", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--diagnostic-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--band-low", type=float, default=50.0)
    parser.add_argument("--band-high", type=float, default=60.0)
    args = parser.parse_args()

    started = time.time()
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    client = VLLMClient(args.base_url, args.model)
    set_a = load_matched_set_a(Path(args.seta_scan), random.Random(args.seed))
    print(f"Loaded matched Set A: {len(set_a)}", flush=True)
    set_b_screen_path = prefix.with_name(prefix.name + "_setb_screen.jsonl")
    set_b, screen_rows = screen_set_b(
        client,
        len(set_a),
        args.workers,
        args.band_low,
        args.band_high,
        set_b_screen_path,
    )
    print(f"Selected matched Set B: {len(set_b)}", flush=True)
    records = run_items(client, set_a, set_b, min(args.workers, len(set_a) + len(set_b)), args.diagnostic_n)
    results_path = prefix.with_name(prefix.name + "_results.jsonl")
    with results_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = extended_summary(
        args.model,
        records,
        screen_rows,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.confidence_level,
    )
    summary["diagnostic_n"] = args.diagnostic_n
    summary["elapsed_seconds"] = time.time() - started
    summary["results_path"] = str(results_path)
    summary["setB_screen_path"] = str(set_b_screen_path)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
