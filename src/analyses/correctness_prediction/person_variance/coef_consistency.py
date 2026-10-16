"""How much the per-participant coefficients differ from one person to the next.

Each participant has a model of their own, trained on their trials alone
(`loo_runs`; "LOO" there means leaving one of that person's trials out at a time,
to score it). The coefficients used here come from one more fit, on all of that
person's trials. Everyone's model has the same few features, so a feature's
coefficient can be compared across people. This module asks three questions:

* **What does the model rely on, on average?** Each feature's mean coefficient
  across participants (`mean_coef_across_participants`), how often it is among a
  person's largest coefficients, and its average rank.
* **Which features mean the same thing for everyone?**
  (`per_feature_coef_consistency`) The share of participants whose coefficient is
  negative, zero or positive, and how widely the coefficients spread compared with
  their typical size.
* **How does one feature's coefficient vary?** (`plot_participant_coef_effect`)
  Every participant's coefficient for it, with its interval.

Two caveats. Each fit standardises its own features, so a coefficient is per
standard deviation of the feature over that person's trials. And each rests on
that person's trials alone (54 on Study 1), so part of the spread is estimation
noise rather than a real difference between people: read the consistency numbers
as an upper bound on how inconsistent a feature is. None of this is a
significance test.
"""

from __future__ import annotations

from typing import Optional, Any, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analyses.correctness_prediction.plots.coefficients import plot_coef_summary_barh
from src.analyses.correctness_prediction.plots.feature_ranks import (
    compute_feature_avg_rank_across_participants,
    plot_top_abs_coef_feature_frequency_across_participants,
    plot_top_features_by_best_avg_rank,
)
from src.analyses.correctness_prediction.person_variance.loo_runs import (
    MODEL_NAME,
    to_nested_results,
)
from src.analyses.correctness_prediction.person_variance.plot_style import (
    NEG_COLOR,
    POS_COLOR,
    ZERO_COLOR,
    clean_feature_labels,
)
from src.lib.plotting.output import save_output


def coef_long_frame(
    results_by_pid: Mapping[str, Any],
    coef_col: str = "coef",
) -> pd.DataFrame:
    """Every participant's coefficient table stacked into one long frame:
    `feature`, `<coef_col>`, `participant_id`.

    Participants without a coefficient table are skipped, and coefficients that are
    not numbers are dropped.
    """
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "feature_presence",
    plot: str = "feature_presence",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """How many participants have each feature among their K largest |coef|.

    K is `top_k_within_participant`; with only a dozen features, 3 keeps the chart
    from collapsing onto one dominant feature. Bars are coloured by the sign most of
    those participants share, and `!n` marks a feature where n of them have the
    other sign. `top_k_features` defaults to every feature.

    Drawn and saved by `plots/feature_ranks.py`, so the files go under
    `reports/correctness_prediction/`, not `.../person_variance/`. Returns
    `(fig, freq_df, saved_paths)`.
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
        subdir=subdir,
        plot=plot, tag=f"{feature_set_tag}_top{top_k_within_participant}_presence",
        to_paper=to_paper,
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "avg_rank",
    plot: str = "avg_rank",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
) -> Tuple[Any, pd.DataFrame, List[str]]:
    """Features ordered by their average rank within each participant's model.

    Rank 1 is that person's largest |coef|, so a low average means the feature is
    consistently among the most important. Drawn and saved by
    `plots/feature_ranks.py`, so the files go under
    `reports/correctness_prediction/`, not `.../person_variance/`.

    Returns `(fig, avg_rank_df, saved_paths)`; `avg_rank_df` is the full ranking
    table, not only the plotted rows.
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
        subdir=subdir,
        plot=plot, tag=f"{feature_set_tag}_mean_ranks",
        to_paper=to_paper,
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
    """Each feature's coefficient averaged over all participants.

    Shaped like a coefficient table, so `plot_coef_summary_barh` can draw it: `coef`
    is the mean, and `ci_low` / `ci_high` are the mean minus and plus one SD across
    participants -- a quick read on how much the effect changes from person to
    person, not a confidence interval. Also has `std` and `n_participants`. Largest
    |mean| first.
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "mean_coef",
    plot: str = "mean_coef",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """Bar chart of `mean_coef_across_participants`, with whiskers one SD either side.

    Drawn and saved by `plots/coefficients.py::plot_coef_summary_barh`, so the files
    go under `reports/correctness_prediction/`, not `.../person_variance/`. Returns
    `(fig, plotted_rows, saved_paths)`.
    """
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
        subdir=subdir,
        plot=plot, tag=f"{feature_set_tag}_mean_coef",
        to_paper=to_paper,
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
    """How consistent each feature's coefficient is across participants.

    Returns `(long_df, summary_df)`: `long_df` has one row per participant and
    feature, `summary_df` one row per feature, most consistent first. In
    `summary_df`:

    - `share_negative` / `share_zero` / `share_positive`: how the participants split
      by sign. A zero coefficient (within `zero_tol`) means the feature had the same
      value on all of that person's trials -- a yes/no feature that never fired,
      say -- so their model could not use it.
    - `dominant_share`: the larger of `share_positive` and `share_negative`. The
      main sort key.
    - `dispersion_index`: the SD of the coefficients divided by their mean absolute
      value. Spread relative to the feature's own typical size, so it compares
      across features: 0 = the same for everyone, above 1 = the spread is larger
      than the typical size. Breaks ties in `dominant_share`, smallest first.
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


# The columns of `per_feature_coef_consistency`'s summary worth printing.
CONSISTENCY_DISPLAY_COLS = [
    "feature", "n_participants", "mean", "share_negative", "share_zero",
    "share_positive", "dominant_share", "dispersion_index",
]


# ---------------------------------------------------------------------------
# Per-participant coefficient, one feature at a time
# ---------------------------------------------------------------------------


def participant_coef_table(
    results_by_pid: Mapping[str, Any],
    feature: str,
    coef_col: str = "coef",
) -> pd.DataFrame:
    """Every participant's coefficient for one feature, smallest first.

    Also carries `se`, `ci_low` and `ci_high` where the coefficient table has them:
    Wald intervals give all three, bootstrap intervals only the bounds, and a run
    without intervals none (NaN). Participants without the feature are left out.
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "participant_coef_effects",
    plot: str = "participant_coef_effects",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
) -> Tuple[Any, pd.DataFrame, List[str]]:
    """Every participant's coefficient for one feature, sorted, with its interval.

    One dot per participant, smallest to largest, with lines at the mean and the
    median. Each whisker is that person's own interval from their full-data fit
    (95% Wald by default), so widths differ from person to person. If anyone lacks
    an interval, every whisker is +/- 1.96 SE instead, and if anyone lacks an SE
    too, none is drawn. The colour follows the sign of the mean.

    Saves `participant_coef_table`'s rows with the figure. Returns
    `(fig, table, saved)`, `saved` being what `save_output` wrote.
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

    saved = save_output(
        fig,
        analysis="correctness_prediction/person_variance",
        save=save,
        subdir=subdir,
        plot=plot,
        tables={"per_participant": d},
        tag=f"{feature_set_tag or 'coef'}_{feature}_participant_effect",
        to_paper=to_paper, dpi=dpi, close=close,
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "coef_consistency",
    plot: str = "coef_consistency",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """Draw `per_feature_coef_consistency`'s result, most consistent feature on top.

    *Left*, the sign split: the share of participants whose coefficient is
    negative, zero or positive, with the share of the more common sign printed
    beside each bar. A nearly solid bar means everyone agrees on the direction; two
    colours mean the effect **changes sign** between people; a wide grey band means
    the feature went unused for many of them.

    *Right*, the spread: each participant's coefficient divided by the feature's
    mean |coef|, so features of different size can be compared and the sign is
    kept. Tight around +/-1 = a similar size for everyone; spread across 0 = it
    varies from person to person. `xlim` keeps a few extreme fits from squashing
    the rest: points beyond it are off the plot, though the boxes still count them.

    Saves the summary table. Returns `(fig, saved)`, `saved` being what
    `save_output` wrote.
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

    saved = save_output(
        fig,
        analysis="correctness_prediction/person_variance",
        save=save,
        subdir=subdir, plot=plot,
        tables={"consistency": summ},
        to_paper=to_paper, dpi=dpi, close=close,
    )
    return fig, saved
