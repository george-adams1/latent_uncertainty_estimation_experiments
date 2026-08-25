# E-C inference-time clustering pilot

Run date: 2026-08-25 UTC

Status: **pilot only; not confirmatory**. The prompt candidate was not frozen at
pilot time; it was subsequently frozen in commit `79554bc` and used unchanged
for the confirmatory run reported in `EC_CLUSTERING_FULL_RESULTS.md`.

## Run configuration

- Clusterer: `meta-llama/Meta-Llama-3-70B-Instruct`
- Revision: `50fd307e57011801c7833c87efa1984ddf2db42f`
- Server: vLLM 0.24.0, temperature 0, seed 0
- Compute: `dgx-26`, all 8 H100s, tensor parallel size 8
- Allocation: Slurm job `91038`
- Pilot size per subject model: 5 Set A + 5 Set B
- Bootstrap: 10,000 item resamples, seed 0, 95% percentile intervals
- Final candidate prompt hashes are stored in both summary JSONs. The lister
  system hash is
  `39e555fabc453588ea8bcf6979d62f3c1eb242d75aab10ba7a329442dc6aac31`.

The machine-readable sources for every number below are:

- `scan_results/ec_clustering_pilot_final_qwen3_8b.jsonl.summary.json`
- `scan_results/ec_clustering_pilot_final_llama3_70b.jsonl.summary.json`

Their per-item records are the same paths without `.summary.json`.
The repeated-run comparison is machine-readable at
`scan_results/ec_clustering_pilot_repeatability.json`.

## Lister behavior

Counts below are items for which the lister returned more than one
interpretation.

| Subject responses | Variant | Set A | Set B |
|---|---|---:|---:|
| Qwen3-8B | Q | 0/5 | 1/5 |
| Qwen3-8B | QS | 4/5 | 1/5 |
| Llama-3-70B | Q | 3/5 | 1/5 |
| Llama-3-70B | QS | 3/5 | 0/5 |

This is the intended Q/QS tradeoff for the Qwen responses: adding samples
substantially increases Set A detection, with a Set B false positive still
present. The Llama-response subset is less clear: QS does not improve the
count-based Set A rate, although it removes the one Set B false positive.
Several multi-entry lists contain descriptions that are arguably semantic
duplicates despite the prompt's deduplication instruction. That is a remaining
clusterer-competence limitation, not a JSON/formatting failure.

## Between-reading variance and prediction

Mean full-sample Set A between-reading variance:

| Subject responses | Oracle | Q | QS | Answer string |
|---|---:|---:|---:|---:|
| Qwen3-8B | 0.0942 | 0.1427 | 0.1483 | 0.2027 |
| Llama-3-70B | 0.1885 | 0.0473 | 0.0473 | 0.2763 |

Set A Pearson correlations with realized clarification gain:

| Subject responses | Split | Oracle | Q | QS | Answer string |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Same 32 | 0.613 | 0.014 | 0.045 | 0.752 |
| Qwen3-8B | First-half predictor | 0.635 | -0.096 | 0.083 | 0.719 |
| Qwen3-8B | Second-half predictor | 0.580 | 0.115 | 0.022 | 0.792 |
| Llama-3-70B | Same 32 | 0.948 | -0.367 | -0.367 | 0.869 |
| Llama-3-70B | First-half predictor | 0.923 | -0.376 | -0.376 | 0.853 |
| Llama-3-70B | Second-half predictor | 0.966 | -0.358 | -0.358 | 0.877 |

These correlations are diagnostics over only five Set A items, not estimates
for the paper. Their bootstrap intervals are correspondingly extreme and are
stored in the summaries. Still, this pilot does **not** show preservation of
the oracle correlation: both inference-time variants collapse toward zero for
Qwen and are negative on this Llama subset. The answer-string correlations are
large here, illustrating why correlation on a five-item selected slice cannot
validate the baseline or the proposed arm.

## Agreement audit

The source alias matcher labels most responses as `other` in this pilot:

- Qwen responses: 411/480 Set A responses
- Llama responses: 372/480 Set A responses

Consequently, all-response agreement is dominated by the handling of `other`.
Among the smaller set of responses uniquely assigned to an annotated reading,
agreement after per-item one-to-one label alignment is:

| Subject responses | Q | QS | Oracle-unique responses |
|---|---:|---:|---:|
| Qwen3-8B | 82.6% | 82.6% | 69 |
| Llama-3-70B | 70.4% | 70.4% | 108 |

Neither variant recovers both oracle reading labels on any of these five items
under the overlap audit. This reflects a combination of under-detection,
assigner collapse, and sparse alias coverage. It should not be read as a clean
semantic-recall estimate.

The Qwen assignment `none` rates on Set A are 17.5% for Q and 35.6% for QS.
Both are 0% for the Llama-response pilot. These model-dependent conventions
materially affect inferred variance and must remain visible in the full report.

## Set B and the answer-string control

Under the approved unchanged E-C definition, Set B has one condition and all
between-reading values are exactly zero. The diagnostic false-positive cost is
therefore visible in lister counts and observed categorical variance rather
than a redefined Set B between term.

For Qwen Set B, mean observed categorical variance is 0.0730 (Q), 0.0871
(QS), and 0.3941 (answer string). This shows the intended answer-string failure:
ordinary surface variation creates much more apparent scatter. All Llama Set B
responses in this pilot collapse to zero observed variance in every arm.

## Prompt development and repeatability

The unfrozen pilot caught and corrected three structural prompt failures before
the final candidate:

1. QS initially combined two readings in one array element.
2. Set B sometimes returned paraphrases of one reading as separate entries.
3. One lister response returned a candidate answer rather than a description of
   what the question requested.

After these corrections, remaining misses and semantic duplicates are treated
as measured Llama-3-70B limitations rather than reasons for further tuning on
the same pilot.

Repeated runs with identical final prompt hashes, seed 0, temperature 0, and
10 concurrent items were not bit-identical under TP8 vLLM:

| Subject responses | Variant | Identical lister lists | Identical assignment labels |
|---|---|---:|---:|
| Qwen3-8B | Q | 9/10 | 96.1% |
| Qwen3-8B | QS | 7/10 | 87.5% |
| Llama-3-70B | Q | 9/10 | 100% |
| Llama-3-70B | QS | 7/10 | 91.0% |

These comparisons use the final-prompt candidate run and the immediately
preceding run with the same prompt hashes. A full run should be treated as one
recorded deterministic-intent realization, or run at lower concurrency if
bit-level repeatability is more important than throughput.

## Assessment before freeze

The implementation and annotation firewall make sense and pass all tests. The
pilot also surfaces the scientific risk the arm was designed to measure:
Llama-3-70B can often describe plausible ambiguity, but its inferred labels do
not preserve the oracle predictor on this tiny subset. The next protocol step
is a human decision: either freeze and commit these prompts, exclude these
pilot IDs, and run the full arm unchanged, or revise the design and start a new
explicitly versioned pilot. No confirmatory full run has been launched.
