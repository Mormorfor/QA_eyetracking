"""What sets apart the participants the model predicts well from those it predicts badly?

Each participant's average of each feature is set against how well their own model
scores them (balanced accuracy by default; see `person_accuracy`), and the
lowest-scoring third of participants is compared with the highest-scoring third.

Two things keep the comparison fair:

* Participants with fewer than `MIN_WRONG_TRIALS` wrong answers are left out. With
  so few wrong answers, balanced accuracy can hardly be estimated and drifts
  toward chance. This also drops those with no scored wrong answer, who have no
  balanced accuracy at all.
* The analysis is meant to be run twice: once on the features the model uses, and
  once on every feature. A difference that shows up only in the second points to
  something the model never sees -- a possible confound -- rather than to a
  weakness in the model's own inputs.

Reading the figures: in the correlation bars, a **positive** bar means people with
more of this are predicted *better* (a model-friendly trait), and a negative bar
the reverse. In the low-vs-high profile the sign means something else: **positive**
means the trait is higher in the people the model does *worst* on.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import src.config.columns as Con
from src.analyses.correctness_prediction.person_variance.plot_style import (
    POS_COLOR,
    signed_bar_colors,
)
from src.lib.plotting.output import save_output

# Fewest wrong answers a participant needs to be kept: with fewer, their balanced
# accuracy rests on too few wrong answers to mean much.
MIN_WRONG_TRIALS = 8

# Which per-person score to characterize; "accuracy" for raw accuracy instead.
ACC_METRIC = "balanced_accuracy"


def per_person_feature_means(
    trial_df: pd.DataFrame,
    characterize_cols: Sequence[str],
    *,
    participant_col: str = Con.PARTICIPANT_ID,
) -> pd.DataFrame:
    """Each participant's average of each feature, over their trials in `trial_df`.

    Values that are not numbers count as missing, and missing values are left out
    of the average. Like any per-person summary, the result depends on which trials
    `trial_df` holds (`docs/pitfalls.md` §3); the full table describes each person
    by all of their trials. `n_trials`, how many trials each person has in
    `trial_df`, is returned beside the averages so a filtered table does not go
    unnoticed: an average over 12 trials looks just like one over 54.
    """
    feat_numeric = trial_df[list(characterize_cols)].apply(pd.to_numeric, errors="coerce")
    out = feat_numeric.groupby(trial_df[participant_col]).mean().reset_index()
    n_trials = (
        trial_df.groupby(participant_col).size().rename("n_trials").reset_index()
    )
    return out.merge(n_trials, on=participant_col, how="left")


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
    save: Optional[bool] = None,
    subdir: Optional[str] = "accuracy_characterization",
    plot: str = "accuracy_characterization",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
) -> Dict[str, Any]:
    """Set each participant's feature averages against their model's score.

    `summary_df` is `person_accuracy.per_person_summary`'s table. Keeps the
    participants with at least `min_wrong_trials` wrong answers, then draws three
    figures, each saved with its numbers:

    1. `class_balance`: the score against the share of answers that were correct,
       and against the share of the rarer outcome;
    2. `feature_corr`: the Spearman correlation of each feature's average with the
       score, strongest `top_n_bars` shown;
    3. `group_profile`: the lowest-scoring third minus the highest-scoring third,
       in standard deviations of each feature across the kept participants (a
       feature that is the same for all of them gets 0).

    Returns a dict: `merged` (scores and feature averages per participant),
    `feat_corr`, `group_cmp`, `figs` and `saved_paths`.
    """
    cols = list(characterize_cols)

    # --- class balance per participant, and the minimum-wrong-answers filter ---
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

    # Both tables have `n_trials`: here it counts the trials the model scored, in
    # the feature means all of the person's trials (`n_trials_all`).
    merged = desc.merge(
        per_person_feature_means(trial_df, cols), on=Con.PARTICIPANT_ID, how="inner",
        suffixes=("", "_all"),
    )

    figs: Dict[str, Any] = {}
    saved_paths: Dict[str, List[str]] = {}

    def _save(key, fig, table):
        figs[key] = fig
        saved_paths[key] = save_output(
            fig,
            analysis="correctness_prediction/person_variance",
            plot=plot,
            tables={"data": table},
            save=save,
            to_paper=to_paper,
            dpi=dpi,
            close=close,
            subdir=subdir,
            label=label or None,
            tag=key,
        ).paths

    # --- 1) accuracy vs class balance -----------------------------------------
    _save("class_balance", _plot_accuracy_vs_class_balance(merged, acc_metric, label), merged)

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
    ), feat_corr)

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
    ), group_cmp.reset_index(names="feature"))

    return {
        "merged": merged,
        "feat_corr": feat_corr,
        "group_cmp": group_cmp,
        "figs": figs,
        "saved_paths": saved_paths,
    }
