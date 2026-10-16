"""Accuracy against reading time on the correct answer, and on the wrong answers.

One figure with two curves: the share of correct answers as a function of the time spent on
the correct answer, and as a function of the average time spent on the three wrong answers.
The times are `RT_normalized_*`: the time spent on visits to each answer, divided by its
number of words (`data-pipeline.md` §7) -- not the per-word area measurements that
`correctness_associations` uses. On Study 1, accuracy hardly changes with time on the correct
answer, and falls steeply with time on the wrong ones.

The binning, the intervals and the drawing are all in the one function below.
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
    Plot the share of correct answers against the reading time on the correct
    answer, and against the mean reading time on the three wrong answers.

    `all_participants` can be the trial table or the word table; it is reduced to
    one row per trial. The times are divided by `rt_divisor` (ms to seconds, by
    default). Each of the two measures is cut into `n_bins` bins of equal width
    over its own range, and each bin gets its accuracy and a 95% Wilson interval.
    `**facets` only go into the file name (e.g. `group="hunters"`): filter the
    frame to a group before calling.

    Things to know when reading the figure:
    - Bins with fewer than `min_n_per_bin` trials are left out of the figure and
      the table.
    - The times are strongly skewed, so most trials fall in the first bin (on
      Study 1, 86% for the correct answer and 97% for the wrong ones) and the
      later points rest on few trials.
    - The intervals treat trials as independent, though each participant
      contributes many trials.
    - Trials missing the correct answer's time, all three wrong answers' times,
      or the correctness value are dropped without a message (values that are
      not numbers count as missing). On Study 1 there are none.

    Saves the figure and its table under `reports/answer_rt_comparison/`.
    Returns `(fig, table)`: one row per curve and bin, with the bin edges and
    centre, the number of trials, how many were correct, the accuracy and the
    interval.
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
