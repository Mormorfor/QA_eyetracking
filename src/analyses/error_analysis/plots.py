"""Figures for the error analysis: which mistakes the model makes, and whether guessing explains them.

Six figures.

Two show what the errors are: a bar chart of the four quadrants for one regime,
and the same chart side by side for all three. The three panels have independent
y-axes, because each regime pools only the trials that match it and the counts
differ several-fold -- compare the shapes, not the heights.

Two show whether the imbalance survives the lucky-guess correction, plotting the
FN/FP ratio for each guessing chance and each of the two ways of placing the
guesses. Here the y-axis is shared, because a ratio has no units and comparing
regimes is the point.

The fifth asks where the two kinds of error come out equal, sweeping the
guessing chance and drawing the two placement rules as a band around it.

The sixth is the confusion matrix drawn as a matrix. Same four numbers as the
bar chart, laid out as what-the-participant-did by what-the-model-said. It is
here rather than in `lib.plotting.confusion` because that one names its axes by
class value (`true_1`, `pred_0`), and what matters here is which cell is which
and that "positive" means the participant got it right.

Every figure saves its numbers beside it; see `conventions.md`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lib.plotting.output import save_output
from src.modeling.confusion import (
    QUAD_COLORS,
    QUAD_LABELS,
    QUAD_ORDER,
    confusion_counts,
    counts_to_frame,
)
from src.analyses.error_analysis.compute import (
    GUESS_MODES,
    GUESS_RATES,
    MODE_LABELS,
    REGIME_TITLES,
    TEST_REGIMES,
    balance_point,
    build_trial_prediction_table,
    confusion_summary,
    fn_fp_ratio,
    guessing_scenarios,
    guessing_scenarios_by_regime,
    quadrants_by_regime,
    ratio_curve,
)

ANALYSIS = "error_analysis"

# What the single-panel figures use unless told otherwise: a reader the model has
# not seen, on an item it has not seen. The strict case the paper leads with.
HEADLINE_REGIME = "test_unseen_subject_unseen_item"

# Colours for the two ways of placing the lucky guesses. Separate from the
# quadrant colours because nothing in those figures is a quadrant.
MODE_COLORS = {"excuse_false_negatives": "#4C72B0", "proportional": "#DD8452"}


# ---------------------------------------------------------------------------
# What the errors are
# ---------------------------------------------------------------------------


def plot_confusion_quadrants(
    trials: pd.DataFrame,
    *,
    regime: str = HEADLINE_REGIME,
    title: Optional[str] = None,
    save: Optional[bool] = None,
    to_paper=None,
    close: Optional[bool] = None,
) -> Tuple[plt.Figure, pd.Series]:
    """Bar chart of the four quadrants for one regime. Returns the figure and the counts."""
    counts = confusion_counts(trials)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(
        [QUAD_LABELS[q] for q in QUAD_ORDER],
        [counts[q] for q in QUAD_ORDER],
        color=[QUAD_COLORS[q] for q in QUAD_ORDER],
    )
    ax.set_ylabel("trials")
    ax.set_title(
        title
        or f"Confusion quadrants - general model\n{REGIME_TITLES.get(regime, regime)} (n={len(trials)})"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="confusion_quadrants",
        tables={"counts": counts_to_frame(counts)},
        save=save,
        to_paper=to_paper,
        close=close,
        regime=regime,
    )
    return fig, counts


def plot_confusion_quadrants_by_regime(
    preds: pd.DataFrame,
    *,
    regimes: Sequence[str] = tuple(TEST_REGIMES),
    save: Optional[bool] = None,
    to_paper=None,
    close: Optional[bool] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """The same bar chart for every test regime, side by side.

    The y-axes are independent on purpose. Each regime pools only its own trials,
    so the counts differ several-fold and one axis would flatten two panels to
    nothing. Compare the shapes.
    """
    summary = quadrants_by_regime(preds, regimes=regimes)

    fig, axes = plt.subplots(1, len(regimes), figsize=(15, 3.8))
    axes = np.atleast_1d(axes)

    for ax, regime in zip(axes, regimes):
        rows = summary[summary["regime"] == regime].set_index("quadrant")
        n = int(rows["n_regime_trials"].iloc[0])
        ax.bar(
            [QUAD_LABELS[q] for q in QUAD_ORDER],
            [int(rows.loc[q, "n_trials"]) for q in QUAD_ORDER],
            color=[QUAD_COLORS[q] for q in QUAD_ORDER],
        )
        ax.set_title(f"{REGIME_TITLES.get(regime, regime)} (n={n})", fontsize=11)
        ax.set_ylabel("trials")
        ax.tick_params(axis="x", rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

    fig.suptitle("Confusion quadrants - general model, by test regime", fontsize=13)
    plt.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="confusion_quadrants_by_regime",
        tables={"counts": summary},
        save=save,
        to_paper=to_paper,
        close=close,
    )
    return fig, summary


def plot_confusion_matrix(
    trials: pd.DataFrame,
    *,
    regime: str = HEADLINE_REGIME,
    normalize: bool = True,
    title: Optional[str] = None,
    save: Optional[bool] = None,
    to_paper=None,
    close: Optional[bool] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """The four quadrants as a matrix: what the participant did by what the model said.

    Rows are the participant's answer, columns the model's, so the diagonal is
    the model getting it right. Each cell shows its quadrant name as well as its
    number.

    `normalize` changes the colour only; both numbers are always printed. It
    defaults to row shares because the counts are so lopsided -- four in five
    answers are correct -- that on a raw scale the three smaller cells are the
    same shade of white. By row, the top is sensitivity and the bottom is
    specificity, which is what compares across regimes of different sizes.
    """
    counts = confusion_counts(trials)

    # Rows: participant correct, participant wrong. Columns: model says correct,
    # model says wrong. Laid out so the quadrant names fall where the docstring
    # in `modeling/confusion.py` draws them.
    grid = np.array(
        [[counts["TP"], counts["FN"]],
         [counts["FP"], counts["TN"]]],
        dtype=float,
    )
    names = np.array([["TP", "FN"], ["FP", "TN"]])

    row_totals = grid.sum(axis=1, keepdims=True)
    shown = grid.copy()
    if normalize:
        safe = np.where(row_totals == 0, 1.0, row_totals)
        shown = grid / safe

    index = ["participant correct", "participant wrong"]
    columns = ["model says correct", "model says wrong"]
    matrix = pd.DataFrame(shown, index=index, columns=columns)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    im = ax.imshow(shown, cmap="Blues")

    ax.set_xticks(range(2), labels=columns)
    ax.set_yticks(range(2), labels=index)
    ax.set_xlabel("what the model predicted")
    ax.set_ylabel("what the participant did")
    ax.set_title(
        title
        or f"Confusion matrix - general model\n{REGIME_TITLES.get(regime, regime)} (n={len(trials)})"
    )

    # Annotate each cell with the quadrant name, the count and the row share, in
    # whichever colour stays readable against the cell.
    threshold = shown.max() / 2 if shown.max() else 0.5
    for i in range(2):
        for j in range(2):
            share = grid[i, j] / row_totals[i, 0] if row_totals[i, 0] else np.nan
            text = (
                f"{names[i, j]}\n{int(grid[i, j]):,}\n{share:.1%} of row"
                if np.isfinite(share)
                else f"{names[i, j]}\n{int(grid[i, j]):,}"
            )
            ax.text(
                j, i, text,
                ha="center", va="center", fontsize=10,
                color="white" if shown[i, j] > threshold else "black",
            )

    fig.colorbar(im, ax=ax, label="row share" if normalize else "trials")
    plt.tight_layout()

    # The saved table always carries the counts, never the normalized view: a
    # share can be recomputed from counts, the reverse needs the totals back.
    table = counts_to_frame(counts)
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="confusion_matrix",
        tables={"counts": table},
        save=save,
        to_paper=to_paper,
        close=close,
        regime=regime,
        scale="normalized" if normalize else "raw",
    )
    return fig, matrix


# ---------------------------------------------------------------------------
# Whether guessing explains them
# ---------------------------------------------------------------------------


def _annotate_clamped(ax, clamped: bool) -> None:
    """Mark a panel where the correction ran out of trials to move.

    On the figure and not only in the table: a clamped bar is not on the same
    footing as the others, and reading it as if it were understates the ratio.
    """
    if clamped:
        ax.text(
            0.99, 0.97, "* estimate exceeds the trials available",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#C44E52",
        )


def plot_guessing_scenarios(
    counts: pd.Series,
    *,
    regime: str = HEADLINE_REGIME,
    rates: Sequence[Tuple[str, float]] = tuple(GUESS_RATES),
    modes: Sequence[str] = tuple(GUESS_MODES),
    save: Optional[bool] = None,
    to_paper=None,
    close: Optional[bool] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FN/FP ratio for each guessing chance and placement rule, against the raw ratio.

    The dashed line is the ratio as observed. A bar below it means guessing
    accounts for part of the imbalance; a bar near 1 means it accounts for
    nearly all of it.
    """
    table = guessing_scenarios(counts, rates=rates, modes=modes, regime=regime)
    raw_ratio = fn_fp_ratio(float(counts["FP"]), float(counts["FN"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(rates))
    offsets = np.linspace(-0.19, 0.19, len(modes)) if len(modes) > 1 else [0.0]

    for mode, off in zip(modes, offsets):
        rows = table[table["mode"] == mode].set_index("scenario")
        ratios = [rows.loc[f"guess {name}", "FN/FP"] for name, _ in rates]
        ratios = [np.nan if not np.isfinite(r) else r for r in ratios]
        bars = ax.bar(
            x + off, ratios, width=0.36,
            color=MODE_COLORS.get(mode, "#888888"), label=MODE_LABELS.get(mode, mode),
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    if np.isfinite(raw_ratio):
        ax.axhline(raw_ratio, ls="--", c="grey", label=f"raw = {raw_ratio:.1f}")
    ax.axhline(1.0, ls=":", c="black", lw=1, label="FN = FP")

    ax.set_xticks(x, labels=[f"p = {name}" for name, _ in rates])
    ax.set_ylabel("FN / FP")
    # The three rates are "ruled out two of the four options, one, none", which
    # is the reading a reader needs to judge whether any of them is plausible.
    ax.set_xlabel("assumed chance a non-knower guesses correctly\n(options ruled out: 2, 1, 0)")
    ax.set_title(
        f"Does guessing explain the FN / FP imbalance?\n"
        f"{REGIME_TITLES.get(regime, regime)}"
    )
    ax.legend(fontsize=8)
    _annotate_clamped(ax, bool(table["clamped"].any()))
    plt.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="guessing_scenarios",
        tables={"scenarios": table},
        save=save,
        to_paper=to_paper,
        close=close,
        regime=regime,
    )
    return fig, table


def plot_guessing_scenarios_by_regime(
    preds: pd.DataFrame,
    *,
    regimes: Sequence[str] = tuple(TEST_REGIMES),
    rates: Sequence[Tuple[str, float]] = tuple(GUESS_RATES),
    modes: Sequence[str] = tuple(GUESS_MODES),
    save: Optional[bool] = None,
    to_paper=None,
    close: Optional[bool] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """The guessing correction for every test regime, on one shared y-axis.

    Shared here, unlike the quadrant panels, because a ratio has no units and
    comparing regimes is the question.
    """
    table = guessing_scenarios_by_regime(preds, regimes=regimes, rates=rates, modes=modes)

    fig, axes = plt.subplots(1, len(regimes), figsize=(16, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(rates))
    offsets = np.linspace(-0.19, 0.19, len(modes)) if len(modes) > 1 else [0.0]

    for ax, regime in zip(axes, regimes):
        sub = table[table["regime"] == regime]
        raw_ratio = float(sub.loc[sub["scenario"] == "raw", "FN/FP"].iloc[0])

        for mode, off in zip(modes, offsets):
            rows = sub[sub["mode"] == mode].set_index("scenario")
            ratios = [rows.loc[f"guess {name}", "FN/FP"] for name, _ in rates]
            ratios = [np.nan if not np.isfinite(r) else r for r in ratios]
            bars = ax.bar(
                x + off, ratios, width=0.36,
                color=MODE_COLORS.get(mode, "#888888"),
                label=MODE_LABELS.get(mode, mode),
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

        if np.isfinite(raw_ratio):
            ax.axhline(raw_ratio, ls="--", c="grey", label=f"raw = {raw_ratio:.1f}")
        ax.axhline(1.0, ls=":", c="black", lw=1)
        ax.set_xticks(x, labels=[f"p = {name}" for name, _ in rates])
        n = int(sub["n_trials"].iloc[0])
        ax.set_title(
            f"{REGIME_TITLES.get(regime, regime)}\n(n={n}, raw FN/FP = {raw_ratio:.1f})",
            fontsize=11,
        )
        ax.set_xlabel("guess-success rate")
        ax.legend(fontsize=7)
        _annotate_clamped(ax, bool(sub["clamped"].any()))

    axes[0].set_ylabel("FN / FP")
    fig.suptitle("Does guessing explain the imbalance? By test regime", fontsize=13)
    plt.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="guessing_scenarios_by_regime",
        tables={"scenarios": table},
        save=save,
        to_paper=to_paper,
        close=close,
    )
    return fig, table


def plot_guessing_balance(
    counts: pd.Series,
    *,
    regime: str = HEADLINE_REGIME,
    p_grid: Optional[Sequence[float]] = None,
    marks: Sequence[Tuple[str, float]] = tuple(GUESS_RATES),
    save: Optional[bool] = None,
    to_paper=None,
    close: Optional[bool] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FN/FP against the guessing chance, with the two placement rules as a band.

    The band is how much the unknown split matters. It closes to a single point
    where the ratio crosses 1, because the gap `FN - FP` loses exactly one trial
    per lucky guess whichever way that guess moves -- so the crossing depends
    only on `p`. See `compute.balance_point`.

    The three chances the scenario tables report are marked on it. Past the
    dotted grey line there are more estimated lucky guesses than false negatives
    to move, so `excuse_false_negatives` runs out and the band stops meaning what
    it means to the left of it.
    """
    curve = ratio_curve(counts, p_grid=p_grid)
    bal = balance_point(counts)
    p_star = bal["p_star"]

    wide = curve.pivot(index="p", columns="mode", values="FN/FP")
    lo = wide.min(axis=1)
    hi = wide.max(axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.fill_between(wide.index, lo, hi, color="#4C72B0", alpha=0.18,
                    label="between the two placement rules")
    for mode in wide.columns:
        ax.plot(wide.index, wide[mode], lw=1.8,
                color=MODE_COLORS.get(mode, "#888888"), label=MODE_LABELS.get(mode, mode))

    ax.axhline(1.0, ls=":", c="black", lw=1)
    ax.text(float(wide.index.max()), 1.0, "FN = FP ", va="bottom", ha="right", fontsize=8)

    if np.isfinite(p_star):
        ax.axvline(p_star, ls="--", c="#C44E52", lw=1.2)
        ax.plot([p_star], [1.0], "o", color="#C44E52", ms=6, zorder=5)
        ax.annotate(
            f"balances at p = {p_star:.3f}\n(needs {bal['n_lucky_needed']:.0f} lucky guesses)",
            xy=(p_star, 1.0), xytext=(p_star - 0.26, 2.4),
            fontsize=9, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1),
        )

    y_top = min(9.0, float(hi.max()) * 1.1)
    for name, p in marks:
        if wide.index.min() <= p <= wide.index.max():
            ax.plot([p, p], [0, y_top], c="grey", lw=0.8, alpha=0.5, zorder=0)
            ax.text(p, y_top * 0.985, f"p = {name}", ha="center", va="top",
                    fontsize=8, color="grey")

    # Past here the estimate wants to move more trials than there are false
    # negatives, so `excuse_false_negatives` runs out and its curve flattens.
    clamped = curve.loc[curve["clamped"], "p"]
    if len(clamped):
        p_clamp = float(clamped.min())
        ax.axvspan(p_clamp, float(wide.index.max()), color="grey", alpha=0.10)
        ax.text(p_clamp, y_top * 0.02, " estimate exceeds\n the false negatives",
                fontsize=7, color="grey", va="bottom", ha="left")

    ax.set_xlabel("assumed chance a non-knower guesses correctly (p)")
    ax.set_ylabel("FN / FP")
    ax.set_ylim(0, y_top)
    ax.set_title(
        f"Where the errors balance\n{REGIME_TITLES.get(regime, regime)}"
    )
    # Lower left: the only corner no curve passes through, and it leaves the
    # p marks along the top readable.
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="guessing_balance",
        tables={"curve": curve, "balance": pd.DataFrame([bal])},
        save=save,
        to_paper=to_paper,
        close=close,
        regime=regime,
    )
    return fig, curve


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_error_analysis(
    preds: pd.DataFrame,
    *,
    regime: str = HEADLINE_REGIME,
    regimes: Sequence[str] = tuple(TEST_REGIMES),
    save: Optional[bool] = None,
    verbose: bool = True,
):
    """Every figure in this analysis, from one table of held-out predictions.

    `regime` is what the single-panel figures use; `regimes` are what the
    side-by-side ones compare. The matrix is drawn once per regime -- its cells
    carry both the count and the row share, so a second, normalized copy would
    only repeat the same table.
    """
    trials = build_trial_prediction_table(preds, regime)
    counts = confusion_counts(trials)

    if verbose:
        print(f"regime = {regime}")
        confusion_summary(trials, verbose=True)

    plot_confusion_quadrants(trials, regime=regime, save=save)
    plot_confusion_quadrants_by_regime(preds, regimes=regimes, save=save)

    for r in regimes:
        plot_confusion_matrix(build_trial_prediction_table(preds, r), regime=r, save=save)

    plot_guessing_scenarios(counts, regime=regime, save=save)
    plot_guessing_scenarios_by_regime(preds, regimes=regimes, save=save)

    for r in regimes:
        plot_guessing_balance(
            confusion_counts(build_trial_prediction_table(preds, r)), regime=r, save=save
        )
