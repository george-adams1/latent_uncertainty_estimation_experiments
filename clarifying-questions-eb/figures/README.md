# Figures

Regenerate with `python3 figures/make_figures.py` (needs matplotlib + numpy).
Each figure is written as `.pdf` for LaTeX and `.png` for review. Every value is
read from a committed artifact — no number is transcribed by hand.

| Figure | Claim it carries | Source |
|---|---|---|
| `fig1_eb_gains` | E-B: at matched 50–60% confidence, clarification pays off on ambiguous items and not on hard unambiguous ones (Llama); Qwen shows oracle value it cannot capture by self-asking | `full_eb_*_summary*.json` → `prediction_1_gain` + its intervals |
| `fig2_ec_between_vs_gain` | E-C: between-reading variance tracks realized clarification gain within Set A | `ec_*_results.jsonl` (per item) + `.summary.json` → `correlations` |
| `fig3_ec_predictors` | E-C prediction 4: the decomposed signal beats undifferentiated variance and scalar confidence | `ec_*_results.jsonl.summary.json` → `correlations` |
| `fig4_ec_decomposition` | E-C: answer variance splits into a removable between-reading part and an irreducible within-reading part; Set B's is zero by construction | `ec_*_results.jsonl.summary.json` → `means` |
| `fig5_diagnostics` | The two caveats: confidence matching leaves a near-binary predictor, and Qwen's Set B null control is not centred on zero | `ec_*_results.jsonl` (per item) |

## Caveats the captions must carry

- **Fig 1.** Llama is leak-filtered (item `9087726812198390660` dropped); the
  script prefers `*_summary_leak_filtered.json` automatically.
- **Fig 2 and 3.** The predictor and the outcome are computed from the same 32
  samples per prompt, so these correlations are not split-sample estimates.
- **Fig 3.** Confidence was restricted to the 50–60% band by design and takes two
  distinct values for Qwen; its weak showing is a property of that restriction.
  The note is drawn on the figure.
- **Fig 5(b).** Set A batches were sampled in the same order as the Set B control
  (ambiguous first, clarified after), so the asymmetry shown here would bias Set A
  gains in the same direction.
