"""Which features matter, across participants: frequency in the top-|coef| set, average
rank, and the feature correlation heatmap.

Split out of the 1,717-line `answer_correctness_viz.py` in stage E (map section 6.4).
"""

# src/predictive_modeling/answer_correctness/answer_correctness_viz.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, List, Sequence, Union, Dict, Any, Tuple, Mapping

import json

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.config.columns as Con
from src.lib.plotting.output import save_output

from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from src.modeling.evaluate import (
    CorrectnessEvaluationResult,
)
from src.lib.plotting.output import save_output


def plot_top_abs_coef_feature_frequency_across_participants(
    results_by_pid: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    coef_col: str = "coef",
    abs_col: Optional[str] = "abs_coef",
    top_k_within_participant: int = 3,
    top_k_features: int = 30,
    min_count: int = 1,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "feature_frequency",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """
    For a given model, compute how often each feature appears in the TOP-K
    absolute coefficients per participant.

    Color encodes the majority sign (positive/negative) among the participant-occurrences.
    If sign varies (both + and - occur), draw an "!" next to the bar.

    Returns (fig, summary_df).
    summary_df columns include: feature, count, prop, n_pos, n_neg, n_zero, majority_sign, sign_varies
    """
    occurrences = []

    for pid, model_dict in results_by_pid.items():
        res = model_dict[model_name]
        coef_df = getattr(res, "coef_summary", None)
        df = coef_df[["feature", coef_col]].dropna().copy()

        df[coef_col] = pd.to_numeric(df[coef_col], errors="coerce")
        df = df.dropna(subset=[coef_col])

        if abs_col is not None and abs_col in coef_df.columns:
            df["_abs"] = pd.to_numeric(coef_df.loc[df.index, abs_col], errors="coerce")
        else:
            df["_abs"] = df[coef_col].abs()
        df = df.dropna(subset=["_abs"])

        topk = df.nlargest(int(top_k_within_participant), "_abs").copy()
        topk = topk.drop_duplicates(subset=["feature"], keep="first")

        for _, row in topk.iterrows():
            c = float(row[coef_col])
            occurrences.append(
                {
                    "participant_id": pid,
                    "feature": row["feature"],
                    "coef": c,
                    "sign": 1 if c > 0 else (-1 if c < 0 else 0),
                }
            )

    occ_df = pd.DataFrame(occurrences)
    total_participants = int(occ_df["participant_id"].nunique())

    agg = (
        occ_df.groupby("feature")
        .agg(
            count=("participant_id", "nunique"),
            n_pos=("sign", lambda s: int((s > 0).sum())),
            n_neg=("sign", lambda s: int((s < 0).sum())),
            n_zero=("sign", lambda s: int((s == 0).sum())),
            total_occ=("sign", "count"),
        )
        .reset_index()
    )

    agg["prop"] = agg["count"] / max(1, total_participants)
    agg["sign_varies"] = (agg["n_pos"] > 0) & (agg["n_neg"] > 0)

    def majority_sign(row) -> int:
        if row["n_pos"] == 0 and row["n_neg"] == 0:
            return 0
        if row["n_pos"] >= row["n_neg"]:
            return 1
        return -1

    agg["majority_sign"] = agg.apply(majority_sign, axis=1)

    def minority_count(row) -> int:
        if row["majority_sign"] > 0:
            return int(row["n_neg"])
        if row["majority_sign"] < 0:
            return int(row["n_pos"])
        return int(min(row["n_pos"], row["n_neg"]))

    agg["minority_count"] = agg.apply(minority_count, axis=1)

    agg = agg[agg["count"] >= int(min_count)].copy()
    agg = agg.sort_values("count", ascending=False).head(int(top_k_features)).copy()
    agg = agg.sort_values("count", ascending=True)

    n_bars = len(agg)
    row_height = 0.32
    min_height = 6
    max_height = 35
    height = min(max(min_height, n_bars * row_height), max_height)

    fig, ax = plt.subplots(figsize=(figsize[0], height))

    y = np.arange(len(agg))
    counts = agg["count"].to_numpy()

    color_pos = "#2ca02c"  # green
    color_neg = "#d62728"  # red
    color_zero = "#7f7f7f"  # gray

    colors = [
        color_pos if s > 0 else (color_neg if s < 0 else color_zero)
        for s in agg["majority_sign"].to_numpy()
    ]

    ax.barh(y, counts, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(agg["feature"].tolist())

    ax.set_xlabel(
        f"# participants where feature is in top {top_k_within_participant} |{coef_col}|"
    )
    ax.set_ylabel("Feature")

    if title is None:
        title = (
            f"Feature presence in top {top_k_within_participant} |{coef_col}| "
            f"across participants – {model_name}"
        )
    ax.set_title(title)

    for i, (_, row) in enumerate(agg.iterrows()):
        if bool(row["sign_varies"]):
            ax.text(
                x=row["count"] + 0.05,
                y=i,
                s=f"!{int(row['minority_count'])}",
                va="center",
                ha="left",
                fontsize=12,
                fontweight="bold",
            )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=color_pos, label="Majority positive"),
        Patch(color=color_neg, label="Majority negative"),
        Patch(color=color_zero, label="No sign / zero"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()

    if filename is None:
        filename = f"topcoef_freq_{model_name}_top{top_k_features}_k{top_k_within_participant}"

    saved_paths = save_output(
        fig,
        analysis="correctness_prediction",
        plot=plot,
        tables={"data": agg},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
        **facets,
    ).paths

    return fig, agg, saved_paths


def compute_feature_avg_rank_across_participants(
    results_by_pid: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    coef_col: str = "coef",
    feature_col: str = "feature",
    abs_col: Optional[str] = "abs_coef",
    rank_method: str = "average",
) -> pd.DataFrame:
    """
    For each participant, rank features by descending |coef| (rank 1 = largest |coef|).
    Aggregate across participants: mean_rank / median_rank / count_participants.

    Returns a DataFrame with:
      feature, count_participants, mean_rank, median_rank, std_rank, min_rank, max_rank
    """
    rows = []

    for pid, per_model in results_by_pid.items():
        res = per_model[model_name]
        coef_df = getattr(res, "coef_summary", None)

        if feature_col not in coef_df.columns or coef_col not in coef_df.columns:
            continue

        df = coef_df[[feature_col, coef_col]].copy()
        df[coef_col] = pd.to_numeric(df[coef_col], errors="coerce")
        df = df.dropna(subset=[coef_col, feature_col])

        # compute abs
        if abs_col is not None and abs_col in coef_df.columns:
            df["_abs"] = pd.to_numeric(coef_df.loc[df.index, abs_col], errors="coerce")
            df["_abs"] = df["_abs"].fillna(df[coef_col].abs())
        else:
            df["_abs"] = df[coef_col].abs()

        df = df.dropna(subset=["_abs"])
        df = df.drop_duplicates(subset=[feature_col], keep="first")

        df["_rank"] = df["_abs"].rank(ascending=False, method=rank_method)

        for _, r in df.iterrows():
            rows.append(
                {
                    "participant_id": pid,
                    "feature": r[feature_col],
                    "rank": float(r["_rank"]),
                    "abs_coef": float(r["_abs"]),
                    "coef": float(r[coef_col]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "count_participants",
                "mean_rank",
                "median_rank",
                "std_rank",
                "min_rank",
                "max_rank",
            ]
        )

    long_df = pd.DataFrame(rows)

    agg = (
        long_df.groupby("feature")
        .agg(
            count_participants=("participant_id", "nunique"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            std_rank=("rank", "std"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .reset_index()
    )
    agg = agg.sort_values(["mean_rank", "count_participants"], ascending=[True, False])

    return agg


def plot_top_features_by_best_avg_rank(
    avg_rank_df: pd.DataFrame,
    top_k: int = 30,
    min_participants: int = 1,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "avg_rank",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """
    Barh plot of the features with the *lowest* mean_rank (best, most consistently high-ranked).
    """
    df = avg_rank_df.copy()
    df = df[df["count_participants"] >= int(min_participants)].copy()
    df = df.nsmallest(int(top_k), "mean_rank").copy()
    df = df.sort_values("mean_rank", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df["feature"], df["mean_rank"])
    ax.invert_yaxis()

    ax.set_xlabel("Mean rank across participants (lower = more consistently important)")
    ax.set_ylabel("Feature")

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if filename is None:
        filename = f"best_avg_rank_top{top_k}_min{min_participants}"

    saved_paths = save_output(
        fig,
        analysis="correctness_prediction",
        plot=plot,
        tables={"data": df},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
        **facets,
    ).paths

    return fig,df, saved_paths


def plot_feature_correlation_heatmap(
    trial_df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    *,
    method: str = "spearman",
    cluster_order: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "feature_correlation",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """
    Correlation heatmap for modeling features.
    """
    if feature_cols is None:
        exclude = {
            Con.PARTICIPANT_ID,
            Con.TRIAL_ID,
            Con.IS_CORRECT_COLUMN,
        }
        numeric_cols = [c for c in trial_df.columns if c not in exclude]
        X = trial_df[numeric_cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.select_dtypes(include=[np.number])
    else:
        cols = [c for c in feature_cols if c in trial_df.columns]
        X = trial_df[cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0.0)

    corr = X.corr(method=method)

    corr_ord = corr
    if cluster_order and corr.shape[0] >= 2:

        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        dist = 1.0 - corr.values
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
        corr_ord = corr.iloc[order, order]


    n = corr_ord.shape[0]

    if figsize is None:
        fig_w = max(10, min(30, 0.35 * n))
        fig_h = max(8, min(30, 0.35 * n))
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_ord.values, vmin=-1, vmax=1, aspect="auto")

    if title is None:
        title = f"Feature correlation ({method})"
        if cluster_order:
            title += " – cluster-ordered"
    ax.set_title(title)

    labels = list(corr_ord.columns)


    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)


    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method.title()} correlation")

    plt.tight_layout()

    if filename is None:
        ord_tag = "clustered" if cluster_order else "plain"
        filename = f"feature_corr_{method}_{ord_tag}_n{n}"

    saved_paths = save_output(
        fig,
        analysis="correctness_prediction",
        plot=plot,
        tables={"data": corr_ord},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
        **facets,
    ).paths

    return fig, corr_ord, saved_paths
