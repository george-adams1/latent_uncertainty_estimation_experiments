"""Regenerate every E-B/E-C figure from the committed run artifacts.

Each figure derives its values from scan_results/*.summary.json (point estimates
and bootstrap intervals) or the raw *_results.jsonl, never from a prose summary.
The source path for each panel is recorded in FIGURES below and echoed to stdout,
so a caption can name the file a number came from.

    python3 figures/make_figures.py

Writes figN_*.pdf (for LaTeX) and figN_*.png (for review) into figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ec import split_half  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCAN = ROOT / "scan_results"
OUT = Path(__file__).resolve().parent

# Light-mode print palette. Slots 1 and 2 of the validated categorical order;
# validated with validate_palette.js (adjacent and all-pairs, light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e8e7e3"
SET_A = "#2a78d6"  # slot 1, blue
SET_B = "#eb6834"  # slot 2, orange
AQUA = "#1baf7a"  # slot 3, used only where rows are directly labelled
MUTED = "#b8b7b2"  # de-emphasis gray, never a data identity

MODELS = [("Qwen3-8B", "qwen3_8b"), ("Llama-3-70B", "llama3_70b")]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",  # never dashed
})


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def eb_summary(slug: str) -> dict:
    """The leak-filtered summary is the reporting summary for Llama."""
    filtered = SCAN / f"full_eb_{slug}_summary_leak_filtered.json"
    return load_json(filtered if filtered.exists() else SCAN / f"full_eb_{slug}_summary.json")


def ec_summary(slug: str) -> dict:
    return load_json(SCAN / f"ec_{slug}_results.jsonl.summary.json")


def ec_records(slug: str) -> list[dict]:
    return load_jsonl(SCAN / f"ec_{slug}_results.jsonl")


def tidy(ax, *, xgrid=False, ticks=True):
    ax.set_axisbelow(True)
    if not ticks:
        ax.tick_params(left=False)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, stem: str, source: str):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf / .png   <- {source}")


def interval(ax, y, point, low, high, color):
    """Dot-and-interval: 2px rule, >=8px marker with a 2px surface ring."""
    ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot([point], [y], "o", color=color, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)


# --------------------------------------------------------------------------
# Figure 1 — E-B clarification gains, the matched-confidence result
# --------------------------------------------------------------------------
def fig1_eb_gains():
    rows = [
        ("Set A  oracle clarification", "setA_oracle_gain", SET_A),
        ("Set A  self-ask", "setA_selfask_gain", SET_A),
        ("Set B  self-ask", "setB_selfask_gain", SET_B),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), sharex=True)
    for ax, (name, slug) in zip(axes, MODELS):
        s = eb_summary(slug)
        ci = s["confidence_intervals"]["prediction_1_gain"]
        for i, (label, key, color) in enumerate(rows):
            y = len(rows) - 1 - i
            interval(ax, y, s["prediction_1_gain"][key] * 100,
                     ci[key]["low"] * 100, ci[key]["high"] * 100, color)
        ax.axvline(0, color=INK_2, linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(name, fontsize=9, color=INK, loc="left", pad=8)
        ax.set_xlabel("accuracy gain (percentage points)")
        tidy(ax, xgrid=True, ticks=False)
    axes[1].tick_params(labelleft=False)
    handles = [plt.Line2D([], [], color=c, ls="none", marker="o", ms=8, mec=SURFACE, mew=2)
               for c in (SET_A, SET_B)]
    fig.legend(handles, ["Set A  (ambiguous)", "Set B  (hard, unambiguous)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Clarification gain at matched 50–60% confidence, with 95% bootstrap intervals",
                 fontsize=9.5, color=INK, x=0.06, ha="left", y=1.06)
    save(fig, "fig1_eb_gains", "scan_results/full_eb_*_summary*.json : prediction_1_gain")


# --------------------------------------------------------------------------
# Figure 2 — E-C: between-reading variance against realized gain (Set A)
# --------------------------------------------------------------------------
def fig2_ec_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, (name, slug) in zip(axes, MODELS):
        recs = [r for r in ec_records(slug) if r["set"] == "A"]
        x = np.array([r["estimator"]["between_reading_variance"] for r in recs])
        y = np.array([r["realized_gain"] for r in recs]) * 100
        ax.plot(x, y, "o", color=SET_A, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, alpha=0.9, zorder=3)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 2)
        ax.plot(grid, slope * grid + intercept, color=SET_A, linewidth=2, alpha=0.35, zorder=2)

        # Split-half is the reported estimate: predictor and outcome come from
        # disjoint halves of the same draws, so they share no sampling noise.
        sh = split_half.analyze(str(SCAN / f"ec_{slug}_results.jsonl"))
        a, b = sh["split_first_half_predictor"], sh["split_second_half_predictor"]
        ax.annotate(
            f"split-half   r = {a['r']:.2f}  [{a['low']:.2f}, {a['high']:.2f}]\n"
            f"                  r = {b['r']:.2f}  [{b['low']:.2f}, {b['high']:.2f}]\n"
            f"same samples r = {sh['same_samples']['r']:.2f}",
            xy=(0.03, 0.96), xycoords="axes fraction", va="top",
            fontsize=7.5, color=INK_2)
        ax.set_title(f"{name}   (n = {len(recs)} Set A items)", fontsize=9, color=INK,
                     loc="left", pad=8)
        ax.set_xlabel("between-reading variance")
        tidy(ax, ticks=(ax is axes[0]))
    axes[0].set_ylabel("realized clarification gain (pp)")
    fig.suptitle("Between-reading variance predicts which ambiguous items benefit from clarification",
                 fontsize=9.5, color=INK, x=0.06, ha="left", y=1.06)
    fig.text(0.06, -0.11,
             "Points plot the full 32-sample estimates. The reported correlation is the split-half one: "
             "the predictor is estimated\nfrom 16 draws per prompt and the outcome from the disjoint 16, "
             "so the two share no sampling noise. Both half-assignments are shown.",
             fontsize=7.5, color=INK_2, ha="left")
    save(fig, "fig2_ec_between_vs_gain", "scan_results/ec_*_results.jsonl (per item) + .summary.json")


# --------------------------------------------------------------------------
# Figure 3 — E-C: predictor comparison
# --------------------------------------------------------------------------
def fig3_ec_predictors():
    rows = [
        ("between-reading variance", "setA_between_variance_vs_gain", SET_A),
        ("undifferentiated answer variance", "setA_observed_variance_vs_gain", SET_B),
        ("scalar confidence", "setA_confidence_vs_gain", AQUA),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), sharex=True)
    for ax, (name, slug) in zip(axes, MODELS):
        s = ec_summary(slug)
        ci = s["confidence_intervals"]["correlations"]
        for i, (label, key, color) in enumerate(rows):
            y = len(rows) - 1 - i
            interval(ax, y, s["correlations"][key], ci[key]["low"], ci[key]["high"], color)
        ax.axvline(0, color=INK_2, linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(-0.75, 1.05)
        ax.set_title(name, fontsize=9, color=INK, loc="left", pad=8)
        ax.set_xlabel("Pearson r with realized gain, Set A only")
        tidy(ax, xgrid=True, ticks=False)
    axes[1].tick_params(labelleft=False)
    fig.text(0.06, -0.14,
             "Confidence was restricted to the 50–60% matching band by design and takes only "
             "two distinct values for Qwen3-8B;\nits weak correlation is a property of that "
             "restriction and is not evidence that confidence is uninformative in general.",
             fontsize=7.5, color=INK_2, ha="left")
    fig.suptitle("Three candidate predictors of clarification gain, with 95% bootstrap intervals",
                 fontsize=9.5, color=INK, x=0.06, ha="left", y=1.06)
    save(fig, "fig3_ec_predictors", "scan_results/ec_*_results.jsonl.summary.json : correlations")


# --------------------------------------------------------------------------
# Figure 4 — E-C: the variance decomposition itself
# --------------------------------------------------------------------------
def fig4_ec_decomposition():
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    labels, within, between = [], [], []
    for name, slug in MODELS:
        means = ec_summary(slug)["means"]
        for st in ("setA", "setB"):
            labels.append(f"{name}\n{'Set A' if st == 'setA' else 'Set B'}")
            within.append(means[st]["within_reading_variance"])
            between.append(means[st]["between_reading_variance"])

    x = np.arange(len(labels))
    gap = 0.004  # surface gap between stacked segments, in data units
    ax.bar(x, within, width=0.42, color=MUTED, zorder=2)
    ax.bar(x, between, width=0.42, bottom=[w + gap for w in within], color=SET_A, zorder=2)
    for xi, w, b in zip(x, within, between):
        if b > 0:
            ax.text(xi, w + b + gap + 0.006, f"{b:.3f}", ha="center", fontsize=8, color=INK)
        else:
            ax.text(xi, w + gap + 0.006, "0 by construction", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean categorical variance")
    ax.set_ylim(0, max(w + b for w, b in zip(within, between)) * 1.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SET_A),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["between-reading (removable by asking)", "within-reading (remains)"],
              loc="upper left", ncol=1)
    ax.set_title("Answer variance splits into a removable and an irreducible part",
                 fontsize=9.5, color=INK, loc="left", pad=10)
    tidy(ax)
    save(fig, "fig4_ec_decomposition", "scan_results/ec_*_results.jsonl.summary.json : means")


# --------------------------------------------------------------------------
# Figure 5 — two diagnostics that qualify the headline numbers
# --------------------------------------------------------------------------
def fig5_diagnostics():
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8))
    fig.subplots_adjust(wspace=0.42)

    # (a) the confidence variable the matching left behind
    ax = axes[0]
    width = 0.38
    per_model = {slug: [r["confidence"] for r in ec_records(slug)] for _, slug in MODELS}
    levels = sorted({v for vals in per_model.values() for v in vals})
    for i, (name, slug) in enumerate(MODELS):
        vals = per_model[slug]
        counts = [vals.count(v) for v in levels]
        ax.bar(np.arange(len(levels)) + (i - 0.5) * (width + 0.02), counts, width=width,
               color=[SET_A, SET_B][i], label=name, zorder=2)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{int(v)}%" for v in levels])
    ax.set_xlabel("verbalized confidence, post-answer")
    ax.set_ylabel("items")
    ax.legend(loc="upper right")
    ax.set_title("(a)  Matching leaves confidence nearly binary", fontsize=9, color=INK,
                 loc="left", pad=8)
    tidy(ax)

    # (b) the Set B null control, which should centre on zero
    ax = axes[1]
    for i, (name, slug) in enumerate(MODELS):
        d = [(r["repeat_accuracy"] - r["original_accuracy"]) * 100
             for r in ec_records(slug) if r["set"] == "B"]
        d = [v for v in d if v != 0]
        row = len(MODELS) - 1 - i
        jitter = np.linspace(-0.09, 0.09, len(d)) if len(d) > 1 else [0]
        ax.plot(d, np.full(len(d), row) + jitter, "o", color=[SET_A, SET_B][i],
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        pos = sum(1 for v in d if v > 0)
        ax.annotate(f"{pos} up / {len(d) - pos} down", xy=(0.98, row + 0.26),
                    xycoords=("axes fraction", "data"), ha="right", fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[0] for m in reversed(MODELS)])
    ax.set_ylim(-0.5, len(MODELS) - 0.5)
    ax.set_xlabel("repeat batch − original batch (pp), identical prompt")
    ax.set_title("(b)  The null control is not centred on zero for Qwen", fontsize=9,
                 color=INK, loc="left", pad=8)
    tidy(ax, xgrid=True, ticks=False)

    save(fig, "fig5_diagnostics", "scan_results/ec_*_results.jsonl (per item)")


FIGURES = [fig1_eb_gains, fig2_ec_scatter, fig3_ec_predictors,
           fig4_ec_decomposition, fig5_diagnostics]

if __name__ == "__main__":
    print("writing figures to", OUT)
    for make in FIGURES:
        make()
