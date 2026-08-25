# E-A Llama-3-70B push instructions

This directory contains the completed E-A freeze-test run against
`meta-llama/Meta-Llama-3-70B-Instruct` on Slurm allocation `93675` (`dgx-02`).
The run used the preregistered `k = 120` configuration and produced all 2,439
expected records with no unparsable responses.

## Files to push

Commit these files relative to the repository root:

```text
freeze_test/PUSH_INSTRUCTIONS.md
freeze_test/README.md
freeze_test/freeze_test.py
freeze_test/results/ea_llama3_70b_93675.run.log
freeze_test/results/freeze_meta-llama/Meta-Llama-3-70B-Instruct_20260825T020545.jsonl
freeze_test/results/llama3_70b_vllm_server_93675.log
```

Do not commit `freeze_test/__pycache__/` or any `*.pyc` files. The unrelated
untracked clustering work under `clarifying-questions-eb/` and
`FOR_GEORGE_clustering.md` is not part of this E-A commit.

The source change adds an `OPENAI_BASE_URL` transport to `call_model`, allowing
E-A to use a local OpenAI-compatible server such as vLLM. It does not change the
preregistered prompts, grid, sample size, thresholds, or scorer.

## Verify after copying

From the repository root, in an environment with NumPy installed:

```bash
python freeze_test/freeze_test.py --selfcheck
wc -l freeze_test/results/freeze_meta-llama/Meta-Llama-3-70B-Instruct_20260825T020545.jsonl
```

The self-check should pass and `wc` should report 2,439 lines. The artifact
SHA-256 checksums from the completed run are:

```text
afffb7f9b2e5b8d969abe6ab98c9e517bd70fe52930750b9c169e5a6e0966375  freeze_test/results/ea_llama3_70b_93675.run.log
c22f8cb658a55cbb98795a9a97dbf585b976268eca3e7d43900e070bf2ef9cd8  freeze_test/results/freeze_meta-llama/Meta-Llama-3-70B-Instruct_20260825T020545.jsonl
e6a0c170d7fb0a2e235b118568981a75323da9830ebc79a9192d9b8a9464ef35  freeze_test/results/llama3_70b_vllm_server_93675.log
```

## Commit and push

After copying `freeze_test/` into a clean clone with write access:

```bash
git add -- \
  freeze_test/PUSH_INSTRUCTIONS.md \
  freeze_test/README.md \
  freeze_test/freeze_test.py \
  freeze_test/results/ea_llama3_70b_93675.run.log \
  freeze_test/results/freeze_meta-llama/Meta-Llama-3-70B-Instruct_20260825T020545.jsonl \
  freeze_test/results/llama3_70b_vllm_server_93675.log
git status --short
git diff --cached --stat
git commit -m "Run E-A with Llama-3-70B"
git push origin main
```

The raw vLLM server log contains carriage-return progress output, so
`git diff --check` may report whitespace warnings for that log. Those warnings
come from the captured vLLM output, not from the source changes.

## Headline result

The registered scorer returned `none` for both the known- and unknown-mixture
arms. In the known-mixture arm, the sampled channel triggered the registered
mode-collapse and goodness-of-fit guards, making the verbalized channel primary.
The verbalized E5 slopes were `1.017` for the known-mixture arm and `-1.284` for
the unknown-mixture arm. Full cell-level output is in the experiment run log.
