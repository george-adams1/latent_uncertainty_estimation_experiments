# E-C inference-time clustering arm

Status: implemented and pilot-tested. The prompts are frozen by the git commit
that contains this protocol; any later prompt edit requires a new version and a
complete rerun.

## Objective

This arm measures how much of E-C's oracle result survives when answer labels
come from a fixed LLM clusterer that never receives AmbigQA annotations. It is
an annotation-blind clustering audit over the saved E-C responses, not a new
subject-model sampling experiment.

The fixed clusterer is
`meta-llama/Meta-Llama-3-70B-Instruct`, revision
`50fd307e57011801c7833c87efa1984ddf2db42f`. The pilot used vLLM 0.24.0,
temperature 0, seed 0, and tensor parallelism over all eight H100s on
`dgx-26`, inside Slurm job `91038`.

## Inputs and annotation firewall

The subject-model inputs are the existing files:

- `scan_results/ec_qwen3_8b_results.jsonl`
- `scan_results/ec_llama3_70b_results.jsonl`

For each Set A item, the source contains 32 ambiguous-question responses and
two batches of 32 responses generated from the two fixed-reading rewrites. For
Set B, it contains two independent 32-response batches of the same question.
No subject-model responses are resampled.

The clusterer receives:

- Q lister: original question only;
- QS lister: original question and the 32 ambiguous/original responses;
- assigner: original question, one inferred interpretation list, and one
  response.

The clusterer never receives gold aliases, annotated readings, fixed-reading
rewrite text, gold answers, the generating condition name, or other responses
during assignment. Annotation-derived labels are read only after all LLM calls
for the agreement audit and unchanged gain calculation.

This makes clustering annotation-blind, but it is not a fully annotation-free
deployment simulation: the two saved Set A response batches were originally
generated using annotated fixed-reading rewrites. A fully deployed variant
would need newly generated clarifications and new subject-model sampling.

## Lister and assigner

The prompts live in `ec/prompts/`. The lister emits one to four descriptions,
with code-assigned IDs `I1` through `I4`. One interpretation is legal. The
assigner emits an interpretation ID or `none`; `none` is used for unrelated,
unclear, or deliberately multi-reading answers. vLLM JSON-schema constrained
decoding enforces the response shape.

Every batch is assigned separately, one response per request. Output records
retain the raw structured response and aligned assignment for every request.
Prompt SHA-256 hashes, clusterer model/revision, serving version, seed, node,
Slurm job/step, source-file SHA-256, and source-file commit are recorded in
every item.

## Estimator

`ec/clustering.py::estimate_from_labels` calls the same
`ec.estimator.decompose_conditionals` function as oracle E-C. For Set A,
assigned labels replace alias labels in the two saved reading-conditioned
response distributions. The categorical statistic remains

`h(p) = 1 - sum_c p(c)^2`,

with a uniform prior over the two generating conditions. The implementation
recomputes the oracle arm through this refactored path and aborts if it differs
from the stored oracle value by more than `1e-12`.

The answer-string arm normalizes responses with the existing E-B normalizer
and gives every distinct normalized string its own cluster.

Set B preserves the original E-C estimand: it has one reading-conditioned
distribution, so between-reading variance is structurally zero for every arm.
The implementation additionally computes an explicitly named
`between_batch_variance` by treating the original and repeat batches as two
null conditions and applying the same decomposition. This measures finite-
sample batch separation, not between-reading variance. Lister counts, observed
categorical variance, and between-batch variance are the Set B false-positive
diagnostics. Keeping the names separate prevents a batch artifact from being
reported as latent reading structure.

## Split-half analysis and audit

For Set A, each arm reports:

- same-sample correlation between full 32-sample variance and full gain;
- first 16 responses for variance versus the second 16 for clarified gain;
- second 16 responses for variance versus the first 16 for clarified gain.

Intervals use the existing deterministic item bootstrap: 10,000 samples,
seed 0, 95% percentile intervals.

Because inferred IDs have no oracle names, agreement uses the one-to-one
mapping from inferred IDs to annotated reading labels that maximizes sample
overlap within each item. This mapping is audit-only. Agreement is reported
both over all responses and over responses uniquely labeled as a reading by
the oracle. Counts of inferred `none`, oracle `multiple`, and oracle `other`
are retained.

## Pilot and prompt-freeze rule

The pilot uses the first five Set A and first five Set B records in each source
file. It is marked `pilot_only_not_confirmatory`. Those IDs must be excluded
from a later full analysis. `--exclude-ids-from` accepts the pilot JSONL as an
exclusion manifest.

The CLI refuses a non-pilot run without `--prompt-commit`. It also reads each
prompt from that commit and verifies byte-for-byte equality with the working
prompt, preventing a nominal prompt hash/commit from masking later edits.

Example full command after prompt approval and commit:

```bash
python -m ec.run_clustering \
  --source scan_results/ec_qwen3_8b_results.jsonl \
  --out scan_results/ec_clustering_qwen3_8b.jsonl \
  --base-url http://127.0.0.1:30093/v1 \
  --clusterer-model meta-llama/Meta-Llama-3-70B-Instruct \
  --clusterer-revision 50fd307e57011801c7833c87efa1984ddf2db42f \
  --subject-model Qwen/Qwen3-8B \
  --prompt-commit PROMPT_FREEZE_COMMIT \
  --exclude-ids-from scan_results/ec_clustering_pilot_final_qwen3_8b.jsonl \
  --workers 10
```

## Implementation files

- `ec/clustering.py`: prompts, lister/assigner, labeling, estimator arms, audit
- `ec/clustering_analysis.py`: summaries and split-half bootstrap
- `ec/run_clustering.py`: structured vLLM client, provenance, resume/exclusion,
  JSONL and summary output
- `ec/prompts/clustering_*.txt`: candidate frozen prompts
- `tests/test_ec_clustering.py`: offline firewall, estimator, Set B, and summary
  tests
