"""Correct-vs-distractor reading-time asymmetry.

Lifted out of `presentation_prep.ipynb` cell 22 in stage F (the notebook is now
`notebooks/exploration/presentation_prep.ipynb`).
It had no home in `src/` and no saved numbers -- `findings.md` section 9.1 calls it *"one of
the most interpretable results in the project"* and notes it "lives in one cell of a
presentation notebook". This is that cell, with its numbers now persisted.

**The result:** accuracy is flat against time spent on the *correct* answer (.81-.85 across
the whole range) and collapses against time spent on the *distractors* (.87 -> ~.36). So the
signal is distractor engagement, not correct-answer engagement.

**Why its own analysis rather than `correctness_associations`** (map section 5.3, settled
2026-09-05): a per-area quantity is a feature, a contrast between per-area quantities is an
analysis. `correctness_associations` is built from the IA metric families; this is built from
the **RT family**, which is run-based and derived from click timestamps. Different input,
different folder.

Compute and plot are **not** separated here. The binning, the Wilson intervals and the drawing
are one function with an inner helper, as written; splitting them would be a rewrite rather
than a move, and the move is what stage F is for.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lib.plotting.output import save_output
from src.lib.stats.proportions import wilson_ci


def plot_correctness_by_answer_a_vs_bcd_mean(
    all_participants: pd.DataFrame,
    answer_a_col: str = "RT_normalized_answer_A",
    answer_b_cols: Tuple[str, str, str] = (
        "RT_normalized_answer_B",
        "RT_normalized_answer_C",
        "RT_normalized_answer_D",
    ),
    correct_col: str = "is_correct",
    participant_col: str = "participant_id",
    trial_col: str = "TRIAL_INDEX",
    n_bins: int = 10,
    min_n_per_bin: int = 5,
    figsize: Tuple[int, int] = (7, 4),
    title: Optional[str] = "Correctness by normalized RT to answer options",
    x_label: Optional[str] = "Normalized RT",
    y_label: Optional[str] = "Proportion correct across trials",
    show_ci: bool = True,
    rt_divisor: float = 1000.0,
    save: Optional[bool] = None,
    to_paper=None,
    **facets,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot correctness rate as a function of:
    1. RT_normalized_answer_A
    2. mean(RT_normalized_answer_B, C, D)

    Deduplicates to one row per participant-trial pair.
    """

    b_col, c_col, d_col = answer_b_cols

    required_cols = [
        participant_col,
        trial_col,
        answer_a_col,
        b_col,
        c_col,
        d_col,
        correct_col,
    ]

    missing_cols = [col for col in required_cols if col not in all_participants.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    def _make_binned_summary(d: pd.DataFrame, x_col: str, label: str) -> pd.DataFrame:
        x = d[x_col].to_numpy()

        if len(x) == 0:
            return pd.DataFrame()

        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))

        if xmin == xmax:
            edges = np.array([xmin - 0.5, xmax + 0.5])
        else:
            edges = np.linspace(xmin, xmax, n_bins + 1)

        bin_idx = np.digitize(d[x_col].to_numpy(), edges, right=False) - 1
        bin_idx[bin_idx == len(edges) - 1] = len(edges) - 2

        valid = (bin_idx >= 0) & (bin_idx < len(edges) - 1)

        tmp = d.loc[valid].copy()
        tmp["_bin"] = bin_idx[valid]

        summary = (
            tmp.groupby("_bin", as_index=False)
            .agg(
                n=(correct_col, "size"),
                k_correct=(correct_col, "sum"),
            )
            .sort_values("_bin")
            .reset_index(drop=True)
        )

        summary["bin_left"] = summary["_bin"].apply(lambda i: float(edges[int(i)]))
        summary["bin_right"] = summary["_bin"].apply(lambda i: float(edges[int(i) + 1]))
        summary["bin_center"] = (summary["bin_left"] + summary["bin_right"]) / 2
        summary["accuracy"] = summary["k_correct"] / summary["n"]

        if min_n_per_bin is not None and min_n_per_bin > 1:
            summary = summary[summary["n"] >= int(min_n_per_bin)].copy()

        cis = summary.apply(
            lambda r: wilson_ci(int(r["k_correct"]), int(r["n"])),
            axis=1,
        )

        summary["ci_low"] = [c[0] for c in cis]
        summary["ci_high"] = [c[1] for c in cis]
        summary["metric"] = label

        return summary

    d = (
        all_participants[required_cols]
        .drop_duplicates(subset=[participant_col, trial_col])
        .copy()
    )

    numeric_cols = [answer_a_col, b_col, c_col, d_col, correct_col]

    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Convert RT columns from ms to seconds
    rt_cols = [answer_a_col, b_col, c_col, d_col]
    d[rt_cols] = d[rt_cols] / rt_divisor

    d["RT_normalized_BCD_mean"] = d[[b_col, c_col, d_col]].mean(axis=1)

    d = d.dropna(
        subset=[answer_a_col, "RT_normalized_BCD_mean", correct_col]
    ).copy()

    d[correct_col] = d[correct_col].astype(int)

    summary_a = _make_binned_summary(
        d=d,
        x_col=answer_a_col,
        label=answer_a_col,
    )

    summary_bcd = _make_binned_summary(
        d=d,
        x_col="RT_normalized_BCD_mean",
        label="Mean of B/C/D",
    )

    summary = pd.concat([summary_a, summary_bcd], ignore_index=True)

    fig, ax = plt.subplots(figsize=figsize)

    for metric, group in summary.groupby("metric"):
        ax.plot(
            group["bin_center"],
            group["accuracy"],
            marker="o",
            label=metric,
        )

        if show_ci and len(group) > 0 and group["ci_low"].notna().any():
            ax.fill_between(
                group["bin_center"].to_numpy(),
                group["ci_low"].to_numpy(),
                group["ci_high"].to_numpy(),
                alpha=0.15,
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = summary[
        [
            "metric",
            "bin_left",
            "bin_right",
            "bin_center",
            "n",
            "k_correct",
            "accuracy",
            "ci_low",
            "ci_high",
        ]
    ]

    # The notebook version returned this frame and dropped it on the floor -- which is the
    # failure `findings.md` exists to document. One row per (metric, bin): the bin edges,
    # how many trials fell in it, and the accuracy with its Wilson interval.
    save_output(
        fig,
        analysis="answer_rt_comparison",
        plot="corr_by_answer_rt",
        tables={"summary": out},
        save=save,
        to_paper=to_paper,
        **facets,
    )

    return fig, out
