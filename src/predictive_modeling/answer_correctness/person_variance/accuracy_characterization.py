"""What characterizes the participants the model reads well (or badly)?

Relate each participant's *mean* feature value to how well their person-specific
model scores them, then contrast the low- against the high-accuracy tercile.

Two guards keep the comparison honest:

* :data:`MIN_WRONG_TRIALS` filters out participants with too few wrong
  (minority-class) trials, where balanced accuracy is essentially unestimable and
  collapses toward chance. This also drops the single-class participants that
  have no balanced accuracy at all.
* The analysis is meant to be run **twice** -- once on only the features the
  model was trained on, once on all available features. Comparing the two
  separates *"the model struggles where its own inputs are weak"* from *"a
  characteristic sits in features the model never sees"* (a potential confound).

How to read the two figures: a **positive** bar in the feature/accuracy
correlation means people who do more of this are people the model predicts
*better* (a model-friendly trait); a negative bar marks a model-hostile trait.
In the low-vs-high profile the sign flips meaning: **positive** = the trait is
elevated in the people the model does *worst* on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import src.constants as Con
from predictive_modeling.answer_correctness.person_variance.plot_style import (
    POS_COLOR,
    signed_bar_colors,
)
from predictive_modeling.common.viz_utils import maybe_save_plot

# Keep participants with at least this many wrong trials, so their score is
# actually estimable.
MIN_WRONG_TRIALS = 8

# Which per-person score to characterize; "accuracy" for raw accuracy instead.
ACC_METRIC = "balanced_accuracy"


def per_person_feature_means(
    trial_df: pd.DataFrame,
    characterize_cols: Sequence[str],
    *,
    participant_col: str = Con.PARTICIPANT_ID,
) -> pd.DataFrame:
    """Mean value of each feature per participant (non-numeric values coerced to
    NaN and skipped)."""
    feat_numeric = trial_df[list(characterize_cols)].apply(pd.to_numeric, errors="coerce")
    return feat_numeric.groupby(trial_df[participant_col]).mean().reset_index()


def _plot_accuracy_vs_class_balance(merged, acc_metric, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, xcol in zip(axes, ["base_rate_correct", "minority_rate"]):
        ax.scatter(merged[xcol], merged[acc_metric], alpha=0.6, color=POS_COLOR)
        ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
        ax.set_xlabel(xcol.replace("_", " "))
        ax.set_ylabel(acc_metric.replace("_", " "))
        ax.set_title(f"{acc_metric.replace('_', ' ')} vs {xcol.replace('_', ' ')}")
        ax.legend()
    fig.suptitle(f"[{label}] accuracy vs class balance", y=1.02, fontsize=13)
    plt.tight_layout()
    return fig


def _plot_signed_barh(series, *, title, xlabel):
    """Horizontal bars coloured by sign, smallest value at the bottom."""
    d = series.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(d) + 1)))
    ax.barh(d.index, d.values, color=signed_bar_colors(d.values))
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def characterize_accuracy(
    summary_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    characterize_cols: Sequence[str],
    *,
    acc_metric: str = ACC_METRIC,
    min_wrong_trials: int = MIN_WRONG_TRIALS,
    label: str = "",
    top_n_bars: int = 25,
    verbose: bool = True,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/accuracy_characterization",
    filename_prefix: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
) -> Dict[str, Any]:
    """Relate per-person mean feature values to per-person model accuracy.

    Filters to participants with >= ``min_wrong_trials`` wrong trials (so the
    score is actually estimable), then draws three things and returns the
    intermediate tables:

    1. accuracy vs class-balance scatters,
    2. per-feature Spearman correlation with accuracy,
    3. low- vs high-accuracy (tercile) standardized feature-profile contrast.

    Returns a dict with ``merged`` (per-person scores + feature means),
    ``feat_corr``, ``group_cmp``, ``figs`` and ``saved_paths``.
    """
    cols = list(characterize_cols)

    # --- per-person descriptors + minority-class filter -----------------------
    desc = summary_df.copy()
    desc["base_rate_correct"] = desc["n_correct_trials"] / desc["n_trials"]
    desc["minority_rate"] = (
        desc[["n_correct_trials", "n_wrong_trials"]].min(axis=1) / desc["n_trials"]
    )
    n_before = len(desc)
    desc = desc[desc["n_wrong_trials"] >= min_wrong_trials].copy()
    if verbose:
        print(
            f"[{label}] {len(desc)}/{n_before} participants kept "
            f"(n_wrong_trials >= {min_wrong_trials}); characterizing '{acc_metric}' "
            f"over {len(cols)} features"
        )

    merged = desc.merge(
        per_person_feature_means(trial_df, cols), on=Con.PARTICIPANT_ID, how="inner"
    )

    figs: Dict[str, Any] = {}
    saved_paths: Dict[str, List[str]] = {}
    prefix = filename_prefix or (label.replace(" ", "_") or "accuracy_characterization")

    def _save(key, fig):
        figs[key] = fig
        saved_paths[key] = maybe_save_plot(
            fig=fig, save=save, rel_dir=rel_dir, filename=f"{prefix}_{key}",
            paper_dirs=paper_dirs, dpi=dpi, close=close,
        )

    # --- 1) accuracy vs class balance -----------------------------------------
    _save("class_balance", _plot_accuracy_vs_class_balance(merged, acc_metric, label))

    # --- 2) per-feature correlation with accuracy -----------------------------
    feat_corr = (
        pd.DataFrame({
            "feature": cols,
            "spearman_r": [
                merged[[acc_metric, f]].dropna().corr(method="spearman").iloc[0, 1]
                for f in cols
            ],
        })
        .assign(abs_r=lambda d: d["spearman_r"].abs())
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )
    _top = (
        feat_corr.dropna(subset=["spearman_r"])
        .head(top_n_bars)
        .set_index("feature")["spearman_r"]
    )
    _save("feature_corr", _plot_signed_barh(
        _top,
        title=f"[{label}] feature levels vs model performance (top {len(_top)})",
        xlabel=f"Spearman corr of per-person mean with {acc_metric.replace('_', ' ')}",
    ))

    # --- 3) low- vs high-accuracy group profile -------------------------------
    q_low, q_high = merged[acc_metric].quantile([1 / 3, 2 / 3])
    low = merged[merged[acc_metric] <= q_low]
    high = merged[merged[acc_metric] >= q_high]
    if verbose:
        print(
            f"[{label}] low: {len(low)} ({acc_metric} <= {q_low:.3f})  |  "
            f"high: {len(high)} ({acc_metric} >= {q_high:.3f})"
        )
        for col in ["base_rate_correct", "minority_rate", "n_trials"]:
            print(f"    {col:18s}: low={low[col].mean():.3f}  high={high[col].mean():.3f}")

    stds = merged[cols].std(ddof=0).replace(0, 1.0)
    z = (merged[cols] - merged[cols].mean()) / stds
    group_cmp = (
        pd.DataFrame({
            "low_acc_mean_z": z.loc[low.index].mean(),
            "high_acc_mean_z": z.loc[high.index].mean(),
        })
        .assign(diff=lambda d: d["low_acc_mean_z"] - d["high_acc_mean_z"])
        .sort_values("diff", key=lambda s: s.abs(), ascending=False)
    )
    _d = group_cmp.dropna(subset=["diff"]).head(top_n_bars)["diff"]
    _save("group_profile", _plot_signed_barh(
        _d,
        title=f"[{label}] feature profile: low- vs high-accuracy (top {len(_d)})",
        xlabel="standardized mean difference: low-acc minus high-acc",
    ))

    return {
        "merged": merged,
        "feat_corr": feat_corr,
        "group_cmp": group_cmp,
        "figs": figs,
        "saved_paths": saved_paths,
    }
