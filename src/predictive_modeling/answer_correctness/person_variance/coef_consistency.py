"""How do the *trained* per-person coefficients vary from person to person?

Every participant's ``coef_summary`` comes from a single logistic-regression fit
on all of that participant's trials, so for a small pinned feature set the
coefficients are directly comparable across people. This module asks two
questions of them:

* **What does the feature set look like on average?** -- mean coefficient per
  feature across participants (:func:`mean_coef_across_participants`), plus the
  two existing cross-participant views (top-|coef| presence and average rank).
* **Which features mean the same thing for everyone?**
  (:func:`per_feature_coef_consistency`) -- the share of participants whose
  coefficient is negative / ~zero / positive, and how far the per-person
  coefficients spread relative to that feature's own typical magnitude.

**Caveat:** each per-person coefficient is fit on only ~50 trials, so part of
the spread is estimation noise rather than true between-person variation --
read the consistency numbers as an *upper bound* on inconsistency. This is a
descriptive read, not a significance test.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictive_modeling.answer_correctness.answer_correctness_viz import (
    compute_feature_avg_rank_across_participants,
    plot_coef_summary_barh,
    plot_top_abs_coef_feature_frequency_across_participants,
    plot_top_features_by_best_avg_rank,
)
from predictive_modeling.answer_correctness.person_variance.loo_runs import (
    MODEL_NAME,
    to_nested_results,
)
from predictive_modeling.answer_correctness.person_variance.plot_style import (
    NEG_COLOR,
    POS_COLOR,
    ZERO_COLOR,
    clean_feature_labels,
)
from predictive_modeling.common.viz_utils import maybe_save_plot


def coef_long_frame(
    results_by_pid: Mapping[str, Any],
    coef_col: str = "coef",
) -> pd.DataFrame:
    """Stack every participant's ``coef_summary`` into one long
    ``feature / <coef_col> / participant_id`` frame (non-numeric coefficients
    dropped). Participants without a coefficient table are skipped."""
    frames = []
    for pid, res in results_by_pid.items():
        cs = getattr(res, "coef_summary", None)
        if cs is None or cs.empty:
            continue
        frames.append(cs[["feature", coef_col]].assign(participant_id=pid))
    if not frames:
        raise ValueError("No participant in `results_by_pid` has a coef_summary.")

    long_df = pd.concat(frames, ignore_index=True)
    long_df[coef_col] = pd.to_numeric(long_df[coef_col], errors="coerce")
    return long_df.dropna(subset=[coef_col])


# ---------------------------------------------------------------------------
# Cross-participant views of the trained coefficients
# ---------------------------------------------------------------------------


def plot_feature_presence_across_participants(
    results_by_pid: Mapping[str, Any],
    *,
    feature_set_tag: str,
    top_k_within_participant: int = 3,
    top_k_features: Optional[int] = None,
    model_name: str = MODEL_NAME,
    title: Optional[str] = None,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/feature_presence",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """How often does each feature land in a participant's top-K |coef|?

    With a small curated feature set, ``top_k_within_participant=3`` keeps the
    chart from collapsing onto a single dominant feature. ``top_k_features``
    defaults to every feature that appears at all. Returns
    ``(fig, freq_df, saved_paths)``.
    """
    nested = to_nested_results(results_by_pid, model_name=model_name)
    if top_k_features is None:
        top_k_features = coef_long_frame(results_by_pid)["feature"].nunique()

    return plot_top_abs_coef_feature_frequency_across_participants(
        results_by_pid=nested,
        model_name=model_name,
        coef_col="coef",
        top_k_within_participant=top_k_within_participant,
        top_k_features=top_k_features,
        title=title or (
            f"Feature presence in top {top_k_within_participant} |coef| across "
            f"participants - {feature_set_tag}"
        ),
        save=save,
        rel_dir=rel_dir,
        filename=filename or f"{feature_set_tag}_top{top_k_within_participant}_presence",
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )


def plot_feature_avg_rank_across_participants(
    results_by_pid: Mapping[str, Any],
    *,
    feature_set_tag: str,
    top_k: Optional[int] = None,
    min_participants: int = 1,
    model_name: str = MODEL_NAME,
    title: Optional[str] = None,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/avg_rank",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
) -> Tuple[Any, pd.DataFrame, List[str]]:
    """Features ordered by their best average within-participant |coef| rank.

    Returns ``(fig, avg_rank_df, saved_paths)`` -- ``avg_rank_df`` is the full
    ranking table, not just the plotted rows.
    """
    nested = to_nested_results(results_by_pid, model_name=model_name)
    avg_rank_df = compute_feature_avg_rank_across_participants(
        results_by_pid=nested,
        model_name=model_name,
        coef_col="coef",
        abs_col="abs_coef",
    )
    if top_k is None:
        top_k = len(avg_rank_df)

    fig, _df_plot, saved = plot_top_features_by_best_avg_rank(
        avg_rank_df,
        top_k=top_k,
        min_participants=min_participants,
        title=title or f"Per-person LOO - features by best average rank - {feature_set_tag}",
        save=save,
        rel_dir=rel_dir,
        filename=filename or f"{feature_set_tag}_mean_ranks",
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )
    return fig, avg_rank_df, saved


# ---------------------------------------------------------------------------
# Mean coefficient across participants
# ---------------------------------------------------------------------------


def mean_coef_across_participants(
    results_by_pid: Mapping[str, Any],
    coef_col: str = "coef",
) -> pd.DataFrame:
    """Average each feature's coefficient across all participants' full fits.

    The returned frame is shaped like a ``coef_summary`` so it can be handed
    straight to :func:`plot_coef_summary_barh`: ``coef`` is the cross-participant
    mean and ``ci_low``/``ci_high`` span mean +/- 1 SD (a quick read on how
    stable each effect is from person to person -- *not* a confidence interval).
    """
    long_df = coef_long_frame(results_by_pid, coef_col=coef_col)

    agg = (
        long_df.groupby("feature")[coef_col]
        .agg(coef="mean", std="std", n_participants="count")
        .reset_index()
    )
    agg["abs_coef"] = agg["coef"].abs()
    agg["ci_low"] = agg["coef"] - agg["std"]
    agg["ci_high"] = agg["coef"] + agg["std"]
    return agg.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def plot_mean_coef_across_participants(
    mean_coef_df: pd.DataFrame,
    *,
    feature_set_tag: str,
    top_k: Optional[int] = None,
    title: Optional[str] = None,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/mean_coef",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Bar plot of :func:`mean_coef_across_participants`; whiskers are +/- 1 SD
    across participants."""
    return plot_coef_summary_barh(
        coef_summary=mean_coef_df,
        value_col="coef",
        top_k=len(mean_coef_df) if top_k is None else top_k,
        significant_only=False,
        title=title or f"Mean per-person coefficient (+/- 1 std) - {feature_set_tag}",
        xlabel="Mean coefficient (log-odds)",
        ylabel="",
        clean_labels=True,
        save=save,
        rel_dir=rel_dir,
        filename=filename or f"{feature_set_tag}_mean_coef",
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )


# ---------------------------------------------------------------------------
# Coefficient consistency across participants
# ---------------------------------------------------------------------------


def per_feature_coef_consistency(
    results_by_pid: Mapping[str, Any],
    coef_col: str = "coef",
    zero_tol: float = 1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-feature consistency of the per-person coefficients.

    Returns ``(long_df, summary_df)`` where ``long_df`` is one row per
    (participant, feature) and ``summary_df`` has one row per feature, sorted
    most- to least-consistent:

    - ``share_negative`` / ``share_zero`` / ``share_positive`` -- direction split
      across participants (a ~zero coefficient means the feature is inactive for
      that person, e.g. it never fires in their trials).
    - ``dominant_share`` = ``max(share_positive, share_negative)`` (sort key).
    - ``dispersion_index`` = ``SD(beta_p) / mean(|beta_p|)`` -- spread relative to
      the feature's own typical magnitude, so it is comparable across features
      (0 = identical for everyone, >1 = spread exceeds the typical size).
    """
    long_df = coef_long_frame(results_by_pid, coef_col=coef_col)

    g = long_df.groupby("feature")[coef_col]
    summ = pd.DataFrame({
        "n_participants": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "std": g.std(ddof=1),
        "mean_abs": g.apply(lambda s: s.abs().mean()),
        "share_positive": g.apply(lambda s: (s > zero_tol).mean()),
        "share_negative": g.apply(lambda s: (s < -zero_tol).mean()),
        "share_zero": g.apply(lambda s: (s.abs() <= zero_tol).mean()),
    })
    summ["dominant_share"] = summ[["share_positive", "share_negative"]].max(axis=1)
    summ["dispersion_index"] = summ["std"] / summ["mean_abs"]
    summ = summ.sort_values(["dominant_share", "dispersion_index"], ascending=[False, True])
    return long_df, summ.reset_index()


CONSISTENCY_DISPLAY_COLS = [
    "feature", "n_participants", "mean", "share_negative", "share_zero",
    "share_positive", "dominant_share", "dispersion_index",
]


# ---------------------------------------------------------------------------
# Per-participant coefficient, one feature at a time
# (mirrors mixed_text_answer_effects.plot_participant_effects' per-term plot)
# ---------------------------------------------------------------------------


def participant_coef_table(
    results_by_pid: Mapping[str, Any],
    feature: str,
    coef_col: str = "coef",
) -> pd.DataFrame:
    """One row per participant with a coefficient for ``feature``: their
    ``coef`` plus ``se`` / ``ci_low`` / ``ci_high`` when the full-data fit's
    ``coef_summary`` carries them (Wald or bootstrap CI), sorted by ``coef``.
    """
    rows = []
    for pid, res in results_by_pid.items():
        cs = getattr(res, "coef_summary", None)
        if cs is None or cs.empty:
            continue
        row = cs.loc[cs["feature"] == feature]
        if row.empty:
            continue
        row = row.iloc[0]
        rows.append({
            "participant_id": pid,
            "coef": pd.to_numeric(row.get(coef_col), errors="coerce"),
            "se": pd.to_numeric(row.get("se"), errors="coerce"),
            "ci_low": pd.to_numeric(row.get("ci_low"), errors="coerce"),
            "ci_high": pd.to_numeric(row.get("ci_high"), errors="coerce"),
        })
    if not rows:
        raise ValueError(f"No participant has a coefficient for feature {feature!r}.")

    d = pd.DataFrame(rows).dropna(subset=["coef"])
    return d.sort_values("coef").reset_index(drop=True)


def plot_participant_coef_effect(
    results_by_pid: Mapping[str, Any],
    feature: str,
    coef_col: str = "coef",
    *,
    feature_set_tag: Optional[str] = None,
    title: Optional[str] = None,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/participant_coef_effects",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
) -> Tuple[Any, pd.DataFrame, List[str]]:
    """Per-participant coefficient for one feature, sorted, with 95% CI whiskers.

    Same layout as the mixed-effects ``plot_participant_effects`` slope plots
    (dots + whiskers sorted low to high, mean/median reference lines), but each
    whisker here is that *participant's own* Wald 95% CI from their full-data
    logistic fit -- not a shared random-effect SD, so widths vary participant to
    participant (and are 0 where no CI was computed, e.g. ``ci_method="none"``).

    Returns ``(fig, table, saved_paths)`` -- ``table`` is
    :func:`participant_coef_table`'s output for this feature.
    """
    d = participant_coef_table(results_by_pid, feature, coef_col=coef_col)

    have_ci = d["ci_low"].notna().all() and d["ci_high"].notna().all()
    if have_ci:
        yerr = np.vstack([
            (d["coef"] - d["ci_low"]).clip(lower=0).to_numpy(),
            (d["ci_high"] - d["coef"]).clip(lower=0).to_numpy(),
        ])
    elif d["se"].notna().all():
        yerr = 1.96 * d["se"].to_numpy()
    else:
        yerr = 0.0

    x = np.arange(len(d))
    color = POS_COLOR if d["coef"].mean() >= 0 else NEG_COLOR
    label = clean_feature_labels([feature])[0]

    fig, ax = plt.subplots(figsize=(12, 3), dpi=dpi)
    ax.errorbar(
        x, d["coef"], yerr=yerr, fmt="none",
        ecolor=color, alpha=0.6, capsize=2, lw=1, zorder=1,
    )
    ax.vlines(x, 0, d["coef"], color=color, alpha=0.4, lw=1, zorder=2)
    ax.scatter(x, d["coef"], color=color, s=14, zorder=3)

    ax.axhline(0, color="black", lw=1)
    ax.axhline(d["coef"].mean(), color="gray", ls="--", lw=1, label="Mean")
    ax.axhline(d["coef"].median(), color="gray", ls="-.", lw=1, label="Median")

    suffix = f" ({feature_set_tag})" if feature_set_tag else ""
    ax.set_title(title or f"Participant-specific coefficient of {label}{suffix}")
    ax.set_xlabel("Participant (sorted)")
    ax.set_ylabel("Coefficient (log-odds)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir,
        filename=filename or f"{feature_set_tag or 'coef'}_{feature}_participant_effect",
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return fig, d, saved


def plot_coef_consistency(
    long_df: pd.DataFrame,
    summ: pd.DataFrame,
    coef_col: str = "coef",
    *,
    xlim: Tuple[float, float] = (-4.5, 4.5),
    seed: int = 0,
    title: str = "Coefficient consistency across participants",
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/coef_consistency",
    filename: str = "coef_consistency",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Two aligned views of :func:`per_feature_coef_consistency`, most
    consistent feature on top.

    *Left* -- direction split: the share of participants whose coefficient is
    negative / ~zero / positive. A near-solid bar means everyone agrees on
    direction; a two-colour split means the effect **flips sign** between people;
    a wide grey band means the feature is **inactive** for many people.

    *Right* -- magnitude spread: each participant's coefficient divided by that
    feature's own ``mean|coef|``, so magnitudes are comparable across features
    while sign and zero-crossing are preserved. Tight around +/-1 = consistent
    magnitude; a cloud straddling 0 = the effect varies person to person.
    ``xlim`` clips a few noisy per-person fits that would otherwise squash the
    bulk of the distribution.

    Returns ``(fig, saved_paths)``.
    """
    feats = summ["feature"].tolist()[::-1]  # most consistent on top
    y = np.arange(len(feats))
    bf = summ.set_index("feature").loc[feats]
    labels = clean_feature_labels(feats)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.5, max(3.5, 0.55 * len(feats) + 1.6)),
        sharey=True, gridspec_kw={"width_ratios": [1.0, 1.5]},
    )

    # --- Left: direction split (negative | ~zero | positive), 100% stacked ----
    sn, sz, sp = (bf["share_negative"].to_numpy(),
                  bf["share_zero"].to_numpy(),
                  bf["share_positive"].to_numpy())
    axL.barh(y, sn, color=NEG_COLOR, label="negative (hurts)")
    axL.barh(y, sz, left=sn, color=ZERO_COLOR, label="~zero / inactive")
    axL.barh(y, sp, left=sn + sz, color=POS_COLOR, label="positive (helps)")
    axL.set_xlim(0, 1)
    axL.set_xlabel("share of participants")
    axL.set_title("Direction split of per-person coefficient")
    for yi, d in zip(y, bf["dominant_share"].to_numpy()):
        axL.text(1.01, yi, f"{d:.0%}", va="center", fontsize=8)
    axL.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3,
               fontsize=8, frameon=False)

    # --- Right: per-person spread, scaled by each feature's own mean|coef| ----
    rng = np.random.default_rng(seed)
    mean_abs = bf["mean_abs"].to_numpy()
    dom_pos = (bf["share_positive"] >= bf["share_negative"]).to_numpy()
    box_colors = [POS_COLOR if p else NEG_COLOR for p in dom_pos]
    box_data = [
        (long_df.loc[long_df["feature"] == f, coef_col].to_numpy() / ma
         if ma > 0 else long_df.loc[long_df["feature"] == f, coef_col].to_numpy())
        for f, ma in zip(feats, mean_abs)
    ]
    axR.axvline(0, color="black", lw=1)
    for xr in (-1, 1):
        axR.axvline(xr, color="grey", ls=":", lw=0.8)
    bp = axR.boxplot(box_data, vert=False, positions=y, widths=0.6,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for yi, v in zip(y, box_data):
        jit = (rng.random(len(v)) - 0.5) * 0.35
        axR.scatter(v, yi + jit, s=6, color="black", alpha=0.22, linewidths=0)
    axR.set_xlim(*xlim)
    axR.set_xlabel("per-person coef  /  feature's mean|coef|\n"
                   "(tight around ±1 = consistent, spread across 0 = varies)")
    axR.set_title("Magnitude spread (scaled per feature)")

    # Set y labels last -- boxplot() on the shared axis resets the tick labels.
    axL.set_yticks(y)
    axL.set_yticklabels(labels)

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()

    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir, filename=filename,
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return fig, saved
