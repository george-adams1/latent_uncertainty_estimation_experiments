"""Run annotation-blind E-C clustering over existing JSONL sample records."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ec.clustering import (
    PROMPT_DIR,
    PROMPT_FILES,
    InferenceClusterer,
    file_sha256,
    run_clustering_item,
)
from ec.clustering_analysis import analyze_clustering


class OpenAICompatibleStructuredClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        seed: int | None = 0,
        timeout: float = 120.0,
        retries: int = 3,
        use_json_schema: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.model = model
        self.seed = seed
        self.timeout = timeout
        self.retries = retries
        self.use_json_schema = use_json_schema

    @staticmethod
    def _extract_json(raw: str) -> dict:
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            start, end = stripped.find("{"), stripped.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(stripped[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("structured response was not a JSON object")
        return payload

    def complete_json(
        self,
        system: str,
        user: str,
        schema_name: str,
        schema: dict,
        max_tokens: int,
    ) -> tuple[dict, str]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.use_json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        encoded = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.url, data=encoded, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.load(response)
                raw = (response_payload["choices"][0]["message"].get("content") or "").strip()
                return self._extract_json(raw), raw
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"structured completion failed after {self.retries + 1} attempts: {error}"
                    ) from error
                time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_source_commit(repo: Path, source: Path) -> str | None:
    try:
        relative = source.resolve().relative_to(repo.resolve())
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(relative)],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (ValueError, OSError, subprocess.CalledProcessError):
        return None


def _verify_prompt_commit(repo: Path, commit: str) -> None:
    """Require the committed prompt bytes to equal the prompts about to run."""
    for filename in PROMPT_FILES.values():
        path = PROMPT_DIR / filename
        relative = path.resolve().relative_to(repo.resolve())
        try:
            committed = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=repo,
                check=True,
                capture_output=True,
            ).stdout
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"could not read {relative} from prompt commit {commit}") from error
        if committed != path.read_bytes():
            raise ValueError(f"working prompt {relative} differs from prompt commit {commit}")


def _get_json(url: str) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)
    except Exception:
        return None


def _server_metadata(base_url: str) -> dict:
    base = base_url.rstrip("/")
    return {
        "version_endpoint": _get_json(base.removesuffix("/v1") + "/version"),
        "models_endpoint": _get_json(base + "/models"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--clusterer-model", required=True)
    parser.add_argument("--clusterer-revision", required=True)
    parser.add_argument("--subject-model", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-per-set", type=int)
    parser.add_argument(
        "--exclude-ids-from",
        action="append",
        default=[],
        type=Path,
        help="JSONL whose (set,id) pairs are excluded, e.g. the marked pilot records",
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--prompt-commit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-json-schema", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.pilot and not args.prompt_commit:
        parser.error("a non-pilot run requires --prompt-commit")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    repo = Path(__file__).resolve().parents[2]
    if args.prompt_commit:
        try:
            _verify_prompt_commit(repo, args.prompt_commit)
        except ValueError as error:
            parser.error(str(error))

    source_rows = _read_jsonl(args.source)
    exclusion_records = []
    excluded_keys = set()
    for exclusion_path in args.exclude_ids_from:
        rows = _read_jsonl(exclusion_path)
        keys = {(row["set"], str(row["id"])) for row in rows}
        excluded_keys.update(keys)
        exclusion_records.append(
            {
                "path": str(exclusion_path),
                "sha256": file_sha256(exclusion_path),
                "n_ids": len(keys),
            }
        )
    source_rows = [
        row for row in source_rows if (row["set"], str(row["id"])) not in excluded_keys
    ]
    if args.limit_per_set is not None:
        selected = []
        for set_name in ("A", "B"):
            selected.extend([row for row in source_rows if row["set"] == set_name][: args.limit_per_set])
        source_rows = selected
    if not source_rows:
        parser.error("source selection is empty")

    client = OpenAICompatibleStructuredClient(
        args.base_url,
        args.clusterer_model,
        seed=args.seed,
        timeout=args.timeout,
        retries=args.retries,
        use_json_schema=not args.no_json_schema,
    )
    clusterer = InferenceClusterer(client)
    source_commit = _git_source_commit(repo, args.source)
    provenance = {
        "subject_model": args.subject_model,
        "clusterer": {
            "model": args.clusterer_model,
            "model_revision": args.clusterer_revision,
            "temperature": 0.0,
            "seed": args.seed,
            "prompt_sha256": clusterer.prompt_hashes,
            "prompt_commit": args.prompt_commit,
            "prompts_frozen": bool(args.prompt_commit),
            "server": _server_metadata(args.base_url),
            "runtime": {
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            },
        },
        "source": {
            "results_path": str(args.source),
            "results_sha256": file_sha256(args.source),
            "results_commit": source_commit,
            "exclusion_manifests": exclusion_records,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.resume and args.out.exists():
        existing = _read_jsonl(args.out)
    completed_keys = {(row["set"], str(row["id"])) for row in existing}
    pending = [row for row in source_rows if (row["set"], str(row["id"])) not in completed_keys]
    records = list(existing)
    mode = "a" if existing else "w"
    with args.out.open(mode) as output, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_clustering_item, clusterer, row, provenance, args.pilot): row
            for row in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            records.append(record)
            print(
                f"clustering items completed={len(existing) + completed}/{len(existing) + len(pending)} "
                f"set={record['set']} id={record['id']}",
                flush=True,
            )

    order = {(row["set"], str(row["id"])): index for index, row in enumerate(source_rows)}
    records.sort(key=lambda row: order[(row["set"], str(row["id"]))])
    with args.out.open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = analyze_clustering(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    summary.update(
        {
            "results_path": str(args.out),
            "results_sha256": file_sha256(args.out),
        }
    )
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
