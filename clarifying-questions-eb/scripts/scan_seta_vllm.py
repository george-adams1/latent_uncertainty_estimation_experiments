#!/usr/bin/env python3
"""Full AmbigQA Set A confidence-band scan via an OpenAI-compatible server."""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eb.data_ambigqa import load_ambigqa_candidates


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

    def complete(self, system: str, user: str, retries: int = 5) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
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


def scan_one(client: VLLMClient, index: int, item, low: float, high: float) -> dict:
    answer = client.complete(ANSWER_SYSTEM, f"Question: {item.ambiguous_question}")
    confidence_response = client.complete(
        CONFIDENCE_SYSTEM,
        f"Question: {item.ambiguous_question}\nProposed answer: {answer}",
    )
    confidence = parse_confidence(confidence_response)
    return {
        "index": index,
        "id": item.id,
        "question": item.ambiguous_question,
        "answer": answer,
        "confidence_response": confidence_response,
        "confidence": confidence,
        "matched": confidence is not None and low <= confidence <= high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--band-low", type=float, default=50.0)
    parser.add_argument("--band-high", type=float, default=60.0)
    args = parser.parse_args()

    items = load_ambigqa_candidates("validation") + load_ambigqa_candidates("train")
    if len(items) != 2956:
        raise RuntimeError(f"Expected exactly 2,956 usable Set A items, got {len(items)}")

    started = time.time()
    client = VLLMClient(args.base_url, args.model)
    records: list[dict | None] = [None] * len(items)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(scan_one, client, i, item, args.band_low, args.band_high): i
            for i, item in enumerate(items)
        }
        for future in as_completed(futures):
            record = future.result()
            records[record["index"]] = record
            completed += 1
            if completed % 100 == 0 or completed == len(items):
                matched = sum(bool(r and r["matched"]) for r in records)
                print(f"completed={completed}/{len(items)} matched_so_far={matched}", flush=True)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    parsed = [r for r in records if r and r["confidence"] is not None]
    matched = [r for r in parsed if r["matched"]]
    summary = {
        "model": args.model,
        "candidate_count": len(items),
        "parsed_confidence_count": len(parsed),
        "unparsed_confidence_count": len(items) - len(parsed),
        "band": [args.band_low, args.band_high],
        "matched_count": len(matched),
        "matched_rate": len(matched) / len(items),
        "elapsed_seconds": time.time() - started,
        "records_path": str(output),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
