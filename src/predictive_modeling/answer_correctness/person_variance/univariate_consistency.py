"""Consistency of *every* feature, via per-person marginal associations.

:mod:`coef_consistency` needs per-person model **coefficients**, so it only
covers the handful of trained features. To ask the same "same for everyone vs.
varies?" question for **every** feature, this module switches to a per-person
**marginal** measure that needs no model: for each participant, the Spearman
correlation of each feature with the outcome over that person's own trials.
Consistency is then how much that association agrees across participants.

Two things to keep in mind when reading the result:

* This is a **marginal** (unadjusted) association, not the model's **partial**
  coefficient. A feature can be consistent on its own yet unstable once the
  other features are held fixed (collinearity), and vice-versa -- so this
  *complements* the trained-feature view, it does not replace it.
* Only participants with enough trials and **both** outcome classes can be
  scored, and a feature that is constant within a person is skipped (tracked as
  ``coverage``). With ~50 trials per person these per-person correlations are
  noisy, so read a ``dominant_share`` near 0.5 as "no reliable direction", not
  as proof of a real split.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.constants as Con
from predictive_modeling.answer_correctness.person_variance.plot_style import (
    NEG_COLOR,
    POS_COLOR,
    clean_feature_labels,
)
from predictive_modeling.common.viz_utils import maybe_save_plot

UNIVARIATE_DISPLAY_COLS = [
    "feature", "n_valid", "coverage", "mean_r", "dominant_share", "std_r", "mean_abs_r",
]


def per_person_univariate_consistency(
    trial_df: pd.DataFrame,
    feature_cols: Sequence[str],
    outcome: str = Con.IS_CORRECT_COLUMN,
    participant_col: str = Con.PARTICIPANT_ID,
    min_trials: int = 20,
    min_class: int = 5,
    method: str = "spearman",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-person MARGINAL association of each feature with the outcome, then its
    consistency across participants -- works for ANY feature (no model needed).

    For each participant with >= ``min_trials`` trials and >= ``min_class`` of
    each outcome class, correlate every feature with ``outcome`` over that
    person's own trials. A constant feature yields NaN and is simply not counted
    for that feature (tracked via ``coverage``).

    Returns ``(rmat, summary_df)``: ``rmat`` is participants x features of
    per-person correlations, ``summary_df`` one row per feature sorted by
    ``mean_abs_r`` (strongest first).
    """
    feature_cols = [c for c in feature_cols if c in trial_df.columns]
    per_pid, n_kept, n_drop = {}, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant feature -> NaN corr, expected
        for pid, g in trial_df.groupby(participant_col):
            y = pd.to_numeric(g[outcome], errors="coerce")
            if (len(g) < min_trials or (y == 1).sum() < min_class
                    or (y == 0).sum() < min_class):
                n_drop += 1
                continue
            X = g[feature_cols].apply(pd.to_numeric, errors="coerce")
            per_pid[pid] = X.corrwith(y, method=method)
            n_kept += 1

    if verbose:
        print(f"participants: kept {n_kept}, dropped {n_drop} "
              f"(need >= {min_trials} trials and >= {min_class} of each class)")
    if not per_pid:
        raise ValueError(
            f"No participant met the filters (>= {min_trials} trials and "
            f">= {min_class} trials of each outcome class)."
        )

    rmat = pd.DataFrame(per_pid).T  # participants x features

    valid = rmat.notna().sum()
    summ = pd.DataFrame({
        "n_valid": valid,
        "coverage": valid / len(rmat),
        "mean_r": rmat.mean(),
        "median_r": rmat.median(),
        "std_r": rmat.std(ddof=1),
        "share_positive": (rmat > 0).sum() / valid,
        "share_negative": (rmat < 0).sum() / valid,
    })
    summ["mean_abs_r"] = rmat.abs().mean()
    summ["dominant_share"] = summ[["share_positive", "share_negative"]].max(axis=1)
    summ = summ.sort_values("mean_abs_r", ascending=False)
    return rmat, summ.reset_index().rename(columns={"index": "feature"})


def most_person_dependent(
    summ: pd.DataFrame,
    n_strongest: int = 60,
) -> pd.DataFrame:
    """Among the ``n_strongest`` features (by ``mean_abs_r``), the ones whose
    direction is *least* consistent across participants -- i.e. where a real
    marginal signal flips sign between people."""
    return summ.head(n_strongest).sort_values("dominant_share")


def plot_univariate_consistency(
    rmat: pd.DataFrame,
    summ_sorted: pd.DataFrame,
    top_k: int = 50,
    title: Optional[str] = None,
    *,
    seed: int = 0,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/univariate_consistency",
    filename: str = "univariate_consistency",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Plot the first ``top_k`` rows of ``summ_sorted`` -- sort it however you
    like first (by ``mean_abs_r`` for the strongest signals,
    ``dominant_share`` via :func:`most_person_dependent` for the least
    consistent ones).

    *Left* is the direction split across participants (with each feature's
    ``coverage`` in brackets), *right* the spread of the per-person
    correlations: tight and off 0 = consistent, straddling 0 = varies.
    Returns ``(fig, saved_paths)``.
    """
    top = summ_sorted.head(top_k)
    feats = top["feature"].tolist()[::-1]
    y = np.arange(len(feats))
    bf = top.set_index("feature").loc[feats]
    labels = clean_feature_labels(feats)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, max(3.5, 0.42 * len(feats) + 1.6)),
        sharey=True, gridspec_kw={"width_ratios": [1.0, 1.4]},
    )
    sn, sp = bf["share_negative"].to_numpy(), bf["share_positive"].to_numpy()
    axL.barh(y, sn, color=NEG_COLOR, label="negative")
    axL.barh(y, sp, left=sn, color=POS_COLOR, label="positive")
    axL.set_xlim(0, 1)
    axL.set_xlabel("share of participants (among valid)")
    axL.set_title("Direction split of per-person association")
    for yi, d, cov in zip(y, bf["dominant_share"], bf["coverage"]):
        axL.text(1.01, yi, f"{d:.0%}  (n {cov:.0%})", va="center", fontsize=7)
    axL.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2,
               fontsize=8, frameon=False)

    rng = np.random.default_rng(seed)
    dom_pos = (bf["share_positive"] >= bf["share_negative"]).to_numpy()
    box_colors = [POS_COLOR if p else NEG_COLOR for p in dom_pos]
    box_data = [rmat[f].dropna().to_numpy() for f in feats]
    axR.axvline(0, color="black", lw=1)
    bp = axR.boxplot(box_data, vert=False, positions=y, widths=0.6,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for yi, v in zip(y, box_data):
        jit = (rng.random(len(v)) - 0.5) * 0.35
        axR.scatter(v, yi + jit, s=6, color="black", alpha=0.2, linewidths=0)
    axR.set_xlim(-1, 1)
    axR.set_xlabel("per-person Spearman(feature, outcome)\n"
                   "(tight & off 0 = consistent, straddling 0 = varies)")
    axR.set_title("Per-person association spread")

    axL.set_yticks(y)          # set last: boxplot() resets shared-axis ticks
    axL.set_yticklabels(labels)
    fig.suptitle(title or f"All-feature consistency (marginal, top {top_k})",
                 y=1.02, fontsize=13)
    fig.tight_layout()

    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir, filename=filename,
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return fig, saved
