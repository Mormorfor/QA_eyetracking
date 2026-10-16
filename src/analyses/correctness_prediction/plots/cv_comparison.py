"""Cross-validation scores by regime: printed tables and a bar chart.

`cv_out` is what the runners in `modeling/crossval.py` return, or what
`load_cross_validation_run` reloads. The numbers come from
`modeling/evaluate.py::summarize_cv_results_by_regime`, which averages each regime's
score over the folds, weighted by how many trials each fold scored; this module only
shows them.

`model_name` must be the name the run gives the model (its key in `model_builders`).
The default is not a name any of the project's runs use.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.modeling.evaluate import summarize_cv_results_by_regime
from src.lib.plotting.output import save_output


def show_cv_results(
    cv_out,
    model_name: Optional[str] = "full_features_correctness_log_reg",
    *,
    metric_col: str = "balanced_accuracy",
) -> Dict[str, pd.DataFrame]:
    """
    Show one model's cross-validation results as five tables, for a notebook: every
    fold and regime; by regime; by regime for the test half only; by regime for the
    validation half only; and one overall line.

    The overall line is a plain average over every fold and regime -- new item, new
    participant and both together -- so it only checks that the run worked. Needs a
    run that scored both halves (`eval_split="both"`, the default): the test-only and
    validation-only tables stop with an error otherwise.

    Returns the five tables in a dict.
    """
    summary_df = cv_out.summary_df.copy()

    if model_name is not None:
        summary_df = summary_df[summary_df["model"] == model_name].copy()

    if metric_col not in summary_df.columns:
        raise ValueError(f"metric_col='{metric_col}' not found in cv_out.summary_df")

    print("=" * 80)
    print(f"CROSS-VALIDATION RESULTS: {model_name}")
    print("=" * 80)

    print("\n1) Fold-level raw results")
    fold_level = summary_df.sort_values(["regime", "fold"]).reset_index(drop=True)
    display(fold_level)

    print(f"\n2) Aggregated by regime ({metric_col})")
    by_regime = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=False,
        val_only=False,
        ci=0.95,
    )
    display(by_regime)

    print(f"\n3) Test-only regimes ({metric_col})")
    test_only = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=True,
        val_only=False,
        ci=0.95,
    )
    display(test_only)

    print(f"\n4) Validation-only regimes ({metric_col})")
    val_only = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=False,
        val_only=True,
        ci=0.95,
    )
    display(val_only)

    print(f"\n5) Overall mean across all fold-regime evaluations ({metric_col})")
    overall = pd.DataFrame([{
        "model": model_name,
        "metric": metric_col,
        "n_rows": len(summary_df),
        "n_folds": summary_df["fold"].nunique(),
        "mean_metric": summary_df[metric_col].mean(),
        "std_metric": summary_df[metric_col].std(),
        "min_metric": summary_df[metric_col].min(),
        "max_metric": summary_df[metric_col].max(),
        "total_n_eval": summary_df["n_eval"].sum(),
    }])
    display(overall)

    return {
        "fold_level": fold_level,
        "by_regime": by_regime,
        "test_only": test_only,
        "val_only": val_only,
        "overall": overall,
    }


def plot_cv_metric_by_regime(
    cv_out,
    model_name: str = "full_features_correctness_log_reg",
    metric_col: str = "balanced_accuracy",
    ci: float = 0.95,
    test_only: bool = False,
    val_only: bool = False,
    figsize: tuple = (10, 6),
    rotate_xticks: int = 30,
    save: Optional[bool] = None,
    to_paper=None,
    subdir: str | None = None,
):
    """
    Bar chart of one model's cross-validated score in each regime, with error bars.

    Each bar is the mean over folds, weighted by trials scored. The error bars are a
    `ci` interval across folds (mean +/- z x standard error, cut to [0, 1]; `ci` must
    be 0.90, 0.95 or 0.99). The folds share most of their training data, so the
    error bars are narrower than the real uncertainty. `test_only` / `val_only` keep
    one half of the held-out trials.

    The figure and the summary table are saved as `cv_metric_by_regime`, named by
    model and metric. Returns the summary table, the figure and the axes.
    """
    summary = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=test_only,
        val_only=val_only,
        ci=ci,
    )

    y = summary["mean_metric"].to_numpy()
    yerr = np.vstack([
        y - summary["ci_low"].to_numpy(),
        summary["ci_high"].to_numpy() - y,
    ])

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        summary["regime"],
        summary["mean_metric"],
        yerr=yerr,
        capsize=6,
    )

    pretty_metric = metric_col.replace("_", " ").title()
    ax.set_ylabel(pretty_metric)
    ax.set_xlabel("Regime")
    ax.set_title(f"{model_name}: mean CV {pretty_metric.lower()} by regime ({int(ci * 100)}% CI)")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    save_output(
        fig,
        analysis="correctness_prediction",
        plot="cv_metric_by_regime",
        tables={"summary": summary},
        save=save,
        to_paper=to_paper,
        subdir=subdir,
        model=model_name,
        metric=metric_col,
    )

    return summary, fig, ax



def plot_cv_metric_by_regime_pretty(
    cv_out,
    model_name: str = "full_features_correctness_log_reg",
    metric_col: str = "balanced_accuracy",
    ci: float = 0.95,
    test_only: bool = False,
    val_only: bool = False,
    figsize: tuple = (10, 6),
):
    """
    `plot_cv_metric_by_regime` with readable regime names on the x-axis.

    The relabelled figure is not saved: the saving happens inside
    `plot_cv_metric_by_regime`, before the labels change, and follows the session
    default (`set_output_defaults`).
    """
    pretty_names = {
        "val_seen_subject_unseen_item": "Val: seen subj,\nunseen item",
        "test_seen_subject_unseen_item": "Test: seen subj,\nunseen item",
        "val_unseen_subject_seen_item": "Val: unseen subj,\nseen item",
        "test_unseen_subject_seen_item": "Test: unseen subj,\nseen item",
        "val_unseen_subject_unseen_item": "Val: unseen subj,\nunseen item",
        "test_unseen_subject_unseen_item": "Test: unseen subj,\nunseen item",
    }

    summary, fig, ax = plot_cv_metric_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        ci=ci,
        test_only=test_only,
        val_only=val_only,
        figsize=figsize,
        rotate_xticks=0,
    )

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(
        [pretty_names.get(r, r) for r in summary["regime"]],
        rotation=0,
        ha="center",
    )
    plt.tight_layout()

    return summary, fig, ax
