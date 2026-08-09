"""Per-person accuracy of the leave-one-trial-out runs.

One row per participant: how many trials they contributed, how those split by
outcome, and how well the person-specific model scored them (raw and balanced
accuracy). Balanced accuracy is ``nan`` for participants whose held-out trials
carry a single outcome class.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import src.constants as Con
from predictive_modeling.answer_correctness.person_variance.plot_style import (
    NEG_COLOR,
    POS_COLOR,
)
from predictive_modeling.common.viz_utils import maybe_save_plot


def per_person_summary(
    results_by_pid: Mapping[str, Any],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per participant: trial counts + leave-one-out accuracies.

    Sorted by balanced accuracy (best first). Returns columns
    ``participant_id, n_trials, n_correct_trials, n_wrong_trials, accuracy,
    balanced_accuracy``.
    """
    rows = []
    for pid, res in results_by_pid.items():
        y_true, y_pred = res.y_true, res.y_pred
        has_both_classes = len(np.unique(y_true)) > 1
        rows.append({
            Con.PARTICIPANT_ID: pid,
            "n_trials": res.n_test,
            "n_correct_trials": res.n_positive,
            "n_wrong_trials": res.n_negative,
            "accuracy": res.accuracy,
            "balanced_accuracy": (
                balanced_accuracy_score(y_true, y_pred) if has_both_classes else np.nan
            ),
        })
    summary_df = (
        pd.DataFrame(rows)
        .sort_values("balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print("Mean per-person accuracy:          ", round(summary_df["accuracy"].mean(), 4))
        print("Mean per-person balanced accuracy: ",
              round(summary_df["balanced_accuracy"].mean(), 4))

    return summary_df


def plot_per_person_accuracy_hist(
    summary_df: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("accuracy", "balanced_accuracy"),
    bins: int = 15,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/accuracy",
    filename: str = "per_person_accuracy_hist",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Distribution of each accuracy metric across participants, with the mean
    and the 0.5 chance line marked. Returns ``(fig, saved_paths)``."""
    colors = [POS_COLOR, NEG_COLOR]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    axes = np.atleast_1d(axes)

    for ax, col, color in zip(axes, metrics, colors * len(metrics)):
        vals = summary_df[col].dropna()
        ax.hist(vals, bins=bins, color=color, edgecolor="white")
        ax.axvline(vals.mean(), color="black", ls="--", lw=1, label=f"mean={vals.mean():.3f}")
        ax.axvline(0.5, color="grey", ls=":", lw=1, label="chance")
        ax.set_title(f"Per-person {col.replace('_', ' ')}")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("participants")
        ax.legend()

    plt.tight_layout()
    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir, filename=filename,
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return fig, saved
