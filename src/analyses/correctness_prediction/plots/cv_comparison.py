"""Cross-validation result figures.

Split out of `answer_correctness/cross_validation.py` in stage D step 7. These
three are the reason that module could not simply become `modeling/crossval.py`:
they import `viz.plot_output` and matplotlib, and `modeling/` imports nothing
from `viz/`. Leaving them in place would have carried that dependency into the
new package and broken the layering rule the restructure is for.

**This is a waypoint, not a destination.** Stage E dissolves `viz/` into
per-analysis `plots.py`, and these belong with the rest of the correctness-model
figures -- `analyses/correctness_prediction/plots/` in the target tree
(`restructure-map.md` §6.4). They sit here so they are in the same queue as every
other `visualisations_*` module rather than in a folder invented ahead of time.

The numbers come from `modeling/evaluate.summarize_cv_results_by_regime`; this
module only renders them.
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
    Display fold-level and aggregated CV results for a model.
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
    Bar plot of mean CV metric by regime, with confidence intervals across folds.
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
    Same as `plot_cv_metric_by_regime`, but with prettier regime labels.
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
