# src/predictive_modeling/answer_correctness/answer_correctness_viz.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union, Dict, Any, Tuple, Mapping

import json

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.constants as Con
from src.predictive_modeling.common.viz_utils import maybe_save_plot

from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from src.predictive_modeling.answer_correctness.evaluation_core import (
    CorrectnessEvaluationResult,
)
from viz.plot_output import save_plot, save_df_csv


def show_correctness_model_results(
    results: Mapping[str, CorrectnessEvaluationResult],
    labels: Iterable[int] = (0, 1),
) -> None:
    """
    Print per-model summary statistics for correctness prediction (is_correct = 0/1),
    including:
      - accuracy
      - balanced accuracy
      - precision / recall / F1 per class
      - macro and weighted averages
      - ROC-AUC / average precision if predicted probabilities are available
    """
    labels = list(labels)

    for model_name, res in results.items():
        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        acc = res.accuracy
        n = res.n_test
        y_true = np.asarray(res.y_true)
        y_pred = np.asarray(res.y_pred)

        print(f"Number of test trials: {n}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
        print(f"Positive (correct) trials: {res.n_positive}")
        print(f"Negative (incorrect) trials: {res.n_negative}")

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        prf_df = pd.DataFrame(
            {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            },
            index=[f"class_{l}" for l in labels],
        )

        print("\nPrecision / Recall / F1 (per class):")
        print(prf_df.to_string(float_format=lambda x: f"{x:.3f}"))

        print(
            "\nAverages:"
            f"\n  macro    P/R/F1: {macro_p:.3f} / {macro_r:.3f} / {macro_f1:.3f}"
            f"\n  weighted P/R/F1: {weighted_p:.3f} / {weighted_r:.3f} / {weighted_f1:.3f}"
        )

        y_prob = getattr(res, "y_prob", None)
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                print(f"\nROC-AUC: {roc_auc:.3f}")
            except Exception:
                pass

            try:
                avg_prec = average_precision_score(y_true, y_prob)
                print(f"Average precision (PR-AUC): {avg_prec:.3f}")
            except Exception:
                pass

        print()




def correctness_results_to_summary_df(
    results: Mapping[str, CorrectnessEvaluationResult],
    labels: Iterable[int] = (0, 1),
    run_identifier: str = "",
    trained_feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    labels = list(labels)
    trained_feature_cols_by_model = trained_feature_cols_by_model or {}

    rows = []
    for model_name, res in results.items():
        y_true = np.asarray(res.y_true)
        y_pred = np.asarray(res.y_pred)
        y_prob = None if getattr(res, "y_prob", None) is None else np.asarray(res.y_prob)

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        # macro / weighted
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )

        bal_acc = balanced_accuracy_score(y_true, y_pred)

        roc_auc = None
        avg_prec = None
        if y_prob is not None:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                roc_auc = None
            try:
                avg_prec = float(average_precision_score(y_true, y_prob))
            except Exception:
                avg_prec = None

        trained_features = list(trained_feature_cols_by_model.get(model_name, []))

        row = {
            "run_identifier": run_identifier,
            "model": model_name,
            "n_test": int(res.n_test),
            "accuracy": float(res.accuracy),
            "balanced_accuracy": float(bal_acc),
            "n_positive": int(res.n_positive),
            "n_negative": int(res.n_negative),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "roc_auc": roc_auc,
            "average_precision": avg_prec,
            "n_features": len(trained_features),
            "trained_feature_cols": " | ".join(trained_features),
        }

        for i, lab in enumerate(labels):
            row[f"precision_class_{lab}"] = float(prec[i])
            row[f"recall_class_{lab}"] = float(rec[i])
            row[f"f1_class_{lab}"] = float(f1[i])
            row[f"support_class_{lab}"] = int(support[i])

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["balanced_accuracy", "accuracy", "macro_f1"], ascending=False)
        .reset_index(drop=True)
    )



def plot_coef_summary_barh(
    coef_summary: pd.DataFrame,
    value_col: Optional[str] = "coef",
    top_k: int = 100,
    title: Optional[str] = None,
    model_name: Optional[str] = None,
    h_or_g: Optional[str] = "all_participants",
    figsize: Optional[Tuple[int, int]] = None,
    save: bool = False,
    rel_dir: str = "answer_correctness/coefficients",
    filename: Optional[str] = None,
    paper_dirs: Optional[list[str]] = None,
    dpi: int = 300,
    close: bool = False,
    significant_only: bool = True,
    significance_eps: float = 0.0,
    # ---- presentation touch-ups (all optional, defaults preserve behavior) ----
    feature_label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = False,
    label_wrap: Optional[int] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_color: Optional[str] = None,
    bar_thickness: Optional[float] = None,
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
):
    """
    Horizontal bar plot of top coefficients (by absolute magnitude), overlay 95% CI error bars.
    Can exclude insignificant.

    Presentation touch-ups (optional, off by default):
        feature_label_map / label_replacements / clean_labels / label_wrap
            tidy the y-axis feature names (see ``_format_comparison_labels``).
        xlabel / ylabel
            axis-label overrides (default: ``value_col`` / "feature").
        bar_color
            single bar color.
        title_fontsize / label_fontsize / tick_fontsize
            font sizes for the title, axis labels and tick labels.
    """

    df = coef_summary.copy()

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if "abs_coef" not in df.columns:
        df["abs_coef"] = df[value_col].abs()
    else:
        df["abs_coef"] = pd.to_numeric(df["abs_coef"], errors="coerce")

    if "ci_low" not in df.columns:
        df["ci_low"] = np.nan
    else:
        df["ci_low"] = pd.to_numeric(df["ci_low"], errors="coerce")

    if "ci_high" not in df.columns:
        df["ci_high"] = np.nan
    else:
        df["ci_high"] = pd.to_numeric(df["ci_high"], errors="coerce")

    if "sig_ci" not in df.columns:
        df["sig_ci"] = False

    sig_mask = (df["ci_low"] > significance_eps) | (df["ci_high"] < -significance_eps)
    df["significant"] = sig_mask

    if significant_only:
        df = df[df["significant"]].copy()

    df = df.sort_values("abs_coef", ascending=False).head(int(top_k)).copy()
    df = df.sort_values(value_col, ascending=True)

    if figsize is not None:
        width, height = figsize
    else:
        # Height scales with the number of bars; few-feature models stay short
        # instead of becoming a single fat block in a tall frame.
        n_bars = len(df)
        row_height = 0.55
        min_height = 1.8
        max_height = 30
        width = 9
        height = min(max(min_height, n_bars * row_height + 1.0), max_height)

    fig, ax = plt.subplots(figsize=(width, height))

    y = np.arange(len(df))
    x = df[value_col].to_numpy()

    barh_kwargs = {}
    if bar_color is not None:
        barh_kwargs["color"] = bar_color
    if bar_thickness is not None:
        barh_kwargs["height"] = bar_thickness

    ax.barh(y, x, **barh_kwargs)
    # Keep a consistent margin around the bars so a single bar isn't flush to
    # the frame edges.
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.axvline(0, linewidth=1)

    display_labels = _format_comparison_labels(
        df["feature"].tolist(),
        label_map=feature_label_map,
        label_replacements=label_replacements,
        clean_labels=clean_labels,
        label_wrap=label_wrap,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels)

    ax.set_xlabel(xlabel if xlabel is not None else value_col,
                  fontsize=label_fontsize)
    ax.set_ylabel(ylabel if ylabel is not None else "feature",
                  fontsize=label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    lo = df["ci_low"].to_numpy()
    hi = df["ci_high"].to_numpy()

    mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(x)
    if mask.any():
        xerr = np.vstack([x[mask] - lo[mask], hi[mask] - x[mask]])
        ax.errorbar(
            x[mask],
            y[mask],
            xerr=xerr,
            fmt="none",
            capsize=2,
            linewidth=1,
            color="black",
            ecolor="black",
        )

    plt.tight_layout()

    if filename is None:
        mn = model_name or "model"
        hg = h_or_g or "group"
        suffix = "_sigonly" if significant_only else ""
        filename = f"{mn}_{hg}_top{top_k}_{value_col}{suffix}"

    saved_paths = maybe_save_plot(
        fig=fig,
        save=save,
        rel_dir=rel_dir,
        filename=filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )
    return fig, df, saved_paths




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
    save: bool = False,
    rel_dir: str = "answer_correctness/feature_frequency",
    filename: Optional[str] = None,
    paper_dirs: Optional[list[str]] = None,
    dpi: int = 300,
    close: bool = False,
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

    saved_paths = maybe_save_plot(
        fig=fig,
        save=save,
        rel_dir=rel_dir,
        filename=filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

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
    save: bool = False,
    rel_dir: str = "answer_correctness/avg_rank",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
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

    saved_paths = maybe_save_plot(
        fig=fig,
        save=save,
        rel_dir=rel_dir,
        filename=filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return fig,df, saved_paths


def plot_feature_correlation_heatmap(
    trial_df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    *,
    method: str = "spearman",
    cluster_order: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save: bool = False,
    rel_dir: str = "answer_correctness/feature_correlation",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
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

    saved_paths = maybe_save_plot(
        fig=fig,
        save=save,
        rel_dir=rel_dir,
        filename=filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return fig, corr_ord, saved_paths





def plot_random_effects_barh(
    random_effects_df: pd.DataFrame,
    id_col: str,
    effect_col: str = "random_intercept",
    title: Optional[str] = None,
    top_n: int = 30,
    sort_by_abs: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save: bool = False,
    rel_dir: Optional[str] = None,
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    df = random_effects_df.copy()
    df = df[[id_col, effect_col]].dropna()

    if sort_by_abs:
        df = df.assign(_abs=df[effect_col].abs()).sort_values("_abs", ascending=False)
    else:
        df = df.sort_values(effect_col, ascending=False)

    df = df.head(top_n).copy()
    df = df.sort_values(effect_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df[id_col].astype(str), df[effect_col])
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Random effect")
    ax.set_ylabel(id_col)
    ax.set_title(title or f"Random effects: {id_col}")

    fig.tight_layout()

    paths = maybe_save_plot(
        fig=fig,
        save=save,
        rel_dir=rel_dir,
        filename=filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return fig, df, paths


def plot_random_effects_distribution(
    random_effects_df: pd.DataFrame,
    effect_col: str = "random_intercept",
    title: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (8, 5),
    save: bool = False,
    rel_dir: Optional[str] = None,
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    vals = pd.to_numeric(random_effects_df[effect_col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=bins)
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Random effect")
    ax.set_ylabel("Count")
    ax.set_title(title or "Random-effects distribution")

    fig.tight_layout()

    paths = maybe_save_plot(
        fig=fig,
        save=save,
        rel_dir=rel_dir,
        filename=filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return fig, vals, paths


def summarize_random_effects(
    random_effects_df: pd.DataFrame,
    group_name: str,
    effect_col: str = "random_intercept",
) -> pd.DataFrame:
    vals = pd.to_numeric(random_effects_df[effect_col], errors="coerce").dropna()

    return pd.DataFrame([{
        "group_name": group_name,
        "n_levels": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "min": float(vals.min()),
        "q25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "q75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
        "mean_abs": float(vals.abs().mean()),
        "max_abs": float(vals.abs().max()),
    }])



def _infer_model_family(model_name: str) -> Optional[str]:
    if not isinstance(model_name, str):
        return None

    name = model_name.lower()

    if "log_reg" in name or "logreg" in name:
        return "logreg"
    if "glmer" in name and "julia" not in name:
        return "glmer"
    if "julia" in name:
        return "julia"

    return None



def collect_correctness_run_reports(
    report_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    filename: str = "model_summary.csv",
    recursive: bool = True,
    sort_by: str = "balanced_accuracy",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Collect correctness run summary CSVs from one or more directories.

    Parameters
    ----------
    report_dirs:
        One folder or a list of folders that contain run report CSVs.
        Example:
            "reports/report_data/answer_correctness/logreg"
            [
                "reports/report_data/answer_correctness/logreg",
                "reports/report_data/answer_correctness/glmer",
                "reports/report_data/answer_correctness/julia",
            ]

    filename:
        CSV filename to search for. Default: "model_summary.csv"

    recursive:
        If True, searches all nested subfolders.

    sort_by:
        Column to sort the final table by.

    ascending:
        Sort direction.

    Returns
    -------
    pd.DataFrame
        Combined dataframe of all found run summaries.
    """
    if isinstance(report_dirs, (str, Path)):
        report_dirs = [report_dirs]

    csv_paths: List[Path] = []

    for folder in report_dirs:
        folder = Path(folder)
        if recursive:
            csv_paths.extend(folder.rglob(filename))
        else:
            csv_paths.extend(folder.glob(filename))

    frames = []
    for csv_path in sorted(set(csv_paths)):

        df = pd.read_csv(csv_path)
        df["source_csv"] = str(csv_path)
        df["source_folder"] = str(csv_path.parent)

        parts = list(csv_path.parts)
        df["model_family"] = df["model"].apply(_infer_model_family)
        frames.append(df)


    out = pd.concat(frames, ignore_index=True)

    preferred_cols = [
        "run_identifier",
        "model_family",
        "model",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "n_test",
        "n_features",
        "trained_feature_cols",
        "source_folder",
    ]
    existing_preferred = [c for c in preferred_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_preferred]
    out = out[existing_preferred + remaining]

    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    return out


def plot_correctness_run_comparison(
    summary_df: pd.DataFrame,
    metric_col: str = "balanced_accuracy",
    label_col: Optional[str] = None,
    top_n: Optional[int] = None,
    figsize: tuple = (12, 8),
    title: Optional[str] = None,
    save: bool = False,
    rel_dir: Optional[str] = None,
    filename: str = "run_comparison_balanced_accuracy",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    ytick_fontsize: Optional[float] = None,
    label_wrap: Optional[int] = None,
    label_split_on_sep: bool = False,
    label_fields: Sequence[str] = ("run_identifier", "model_family", "n_features"),
    clean_labels: bool = True,
    label_replacements: Optional[Mapping[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    value_fmt: str = "{:.3f}",
    show_values: bool = True,
):
    """
    Create a horizontal bar plot comparing runs by balanced accuracy.

    Parameters
    ----------
    summary_df:
        Combined dataframe returned by collect_correctness_run_reports().

    metric_col:
        Metric to plot. Default: balanced_accuracy

    label_col:
        Column to use as bar labels.
        If None, a readable label is constructed automatically.

    top_n:
        Optionally only plot the top N runs.

    ytick_fontsize:
        Font size for the y-axis (run label) tick text. If None, uses the
        matplotlib default.

    label_wrap:
        If set, wrap each y-axis label to this maximum character width,
        breaking onto multiple lines.

    label_split_on_sep:
        If True, put each "|"-separated part of the label on its own line.
        Takes precedence over label_wrap when the label contains "|".

    label_fields:
        Which fields (and in what order) make up the auto-generated label
        when label_col is None. Choose from "run_identifier", "model_family",
        "n_features". E.g. ("run_identifier", "n_features") drops the model
        family (the "logreg" part).

    clean_labels:
        If True (default), tidy y-axis labels for presentation: apply
        ``label_replacements`` then replace remaining underscores with spaces
        and collapse repeated whitespace.

    label_replacements:
        Optional {find: replace} mapping applied to each label before the
        underscore cleanup. Use it for full control over wording, e.g.
        {"correct+mean_wrong RT": "Correct vs. mean-wrong RT", "RT": "reaction time"}.

    xlabel / ylabel:
        Axis-label overrides for presentation. If None, the x-axis is derived
        from ``metric_col`` (underscores -> spaces, title-cased) and the y-axis
        is left blank (the per-bar labels are self-explanatory). Pass an empty
        string to force a blank axis label.

    value_fmt:
        Format string for the per-bar value annotations. Default "{:.3f}".

    show_values:
        If True (default), annotate each bar with its metric value.

    Returns
    -------
    fig, plot_df, saved_paths
    """

    df = summary_df.copy()

    if label_col is None:
        def make_label(row):
            field_parts = {
                "run_identifier": (
                    row["run_identifier"]
                    if "run_identifier" in row and pd.notna(row["run_identifier"]) and str(row["run_identifier"]).strip()
                    else None
                ),
                "model_family": (
                    row["model_family"]
                    if "model_family" in row and pd.notna(row["model_family"])
                    else None
                ),
                "n_features": (
                    str(row["n_features"]) + " features"
                    if "n_features" in row and pd.notna(row["n_features"])
                    else None
                ),
            }

            parts = [field_parts[f] for f in label_fields if field_parts.get(f)]
            return " | ".join(parts)

        df["_plot_label"] = df.apply(make_label, axis=1)
        label_col = "_plot_label"

    df = df.dropna(subset=[metric_col]).copy()
    df = df.sort_values(metric_col, ascending=False)

    if top_n is not None:
        df = df.head(top_n).copy()

    df = df.sort_values(metric_col, ascending=True)

    import re

    def clean_label(text: str) -> str:
        text = str(text)
        if label_replacements:
            for find, replace in label_replacements.items():
                text = text.replace(find, replace)
        if clean_labels:
            text = text.replace("_", " ")
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def wrap_label(text: str) -> str:
        text = clean_label(text)
        if label_split_on_sep and "|" in text:
            return "\n".join(part.strip() for part in text.split("|"))
        if label_wrap is not None:
            import textwrap
            return "\n".join(textwrap.wrap(text, width=label_wrap)) or text
        return text

    df = df.copy()
    df[label_col] = df[label_col].map(wrap_label)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df[label_col].astype(str), df[metric_col])
    ax.set_xlabel(xlabel if xlabel is not None else metric_col.replace("_", " ").title())
    ax.set_ylabel(ylabel if ylabel is not None else "")
    ax.set_title(title or f"Comparison of runs by {metric_col.replace('_', ' ')}")

    if ytick_fontsize is not None:
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

    if show_values:
        # Extend the x-axis so the value annotations don't spill past the frame.
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.08)
        for i, val in enumerate(df[metric_col]):
            ax.text(val, i, " " + value_fmt.format(val), va="center")

    plt.tight_layout()

    saved_paths = []
    if save:
        saved_paths = save_plot(
            fig=fig,
            rel_dir=rel_dir,
            filename=filename,
            dpi=dpi,
            paper_dirs=paper_dirs,
            close=close,
        )
    elif close:
        plt.close(fig)

    return fig, df, saved_paths


# ---------------------------------------------------------------------
# Staged cross-validation model comparison (presentation bar chart)
# ---------------------------------------------------------------------

def _format_comparison_labels(
    models: Sequence[str],
    *,
    label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = True,
    label_wrap: Optional[int] = None,
) -> List[str]:
    """Turn raw model identifiers into presentation-ready bar labels.

    Order of operations per label: explicit ``label_map`` lookup (falls back to
    the raw identifier) -> ``label_replacements`` substring swaps -> underscore
    cleanup (if ``clean_labels``) -> optional wrapping to ``label_wrap`` chars.
    """
    import re
    import textwrap

    out: List[str] = []
    for model in models:
        text = str(label_map[model]) if (label_map and model in label_map) else str(model)
        if label_replacements:
            for find, replace in label_replacements.items():
                text = text.replace(find, replace)
        if clean_labels:
            text = text.replace("_", " ")
            text = re.sub(r"\s+", " ", text).strip()
        if label_wrap is not None:
            text = "\n".join(textwrap.wrap(text, width=label_wrap)) or text
        out.append(text)
    return out


def plot_cv_model_comparison_staged(
    comparison_df: pd.DataFrame,
    *,
    model_col: str = "model",
    mean_col: str = "mean_metric",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    # ---- labels --------------------------------------------------------
    label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = True,
    label_wrap: Optional[int] = None,
    # ---- figure / axes -------------------------------------------------
    figsize: Tuple[float, float] = (9.0, 6.0),
    xlim: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = "Balanced accuracy",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    stage_titles: Optional[Sequence[str]] = None,
    # ---- fonts ---------------------------------------------------------
    title_fontsize: float = 16,
    xlabel_fontsize: float = 14,
    ylabel_fontsize: float = 14,
    tick_fontsize: float = 12,
    value_fontsize: float = 12,
    # ---- bars ----------------------------------------------------------
    base_color: str = "#4C72B0",
    highlight_color: str = "#DD8452",
    bar_height: float = 0.62,
    bar_edgecolor: str = "none",
    bar_alpha: float = 1.0,
    # ---- error bars (CIs) ---------------------------------------------
    show_ci: bool = True,
    capsize: float = 4,
    error_color: str = "#333333",
    error_linewidth: float = 1.4,
    # ---- value annotations --------------------------------------------
    show_values: bool = True,
    value_fmt: str = "{:.3f}",
    value_pad: float = 0.006,
    annotate_ci: bool = False,
    ci_fmt: str = "[{:.3f}, {:.3f}]",
    # ---- pending (not-yet-revealed) styling ---------------------------
    show_all_labels: bool = True,
    revealed_label_color: str = "#000000",
    pending_label_color: str = "#BBBBBB",
    # ---- reference (chance) line --------------------------------------
    chance_line: Optional[float] = None,
    chance_label: Optional[str] = None,
    chance_label_pos: str = "bottom",
    chance_color: str = "#888888",
    chance_linestyle: str = "--",
    # ---- grid ----------------------------------------------------------
    grid: bool = True,
    grid_axis: str = "x",
    grid_alpha: float = 0.4,
    # ---- staging / saving ---------------------------------------------
    n_stages: Optional[int] = None,
    save: bool = False,
    rel_dir: str = "answer_correctness/cross_val_comp",
    filename: str = "cross_val_comparison",
    regime: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
) -> Dict[str, Any]:
    """
    Build a horizontal bar chart comparing models by a cross-validated metric
    (with confidence intervals), revealed in stages.

    The models are sorted low -> high and one bar is revealed per stage, from
    lowest to highest. The newly added bar is drawn in ``highlight_color`` while
    previously revealed bars use ``base_color``. Every stage uses identical
    figure size, axis limits and y-tick labels (all category labels are always
    shown when ``show_all_labels``), so the saved frames do not jump between
    stages -- suitable for an animated/click-through conference slide.

    Parameters
    ----------
    comparison_df:
        One row per model with the mean metric and CI bounds, e.g. the output of
        ``build_cv_model_comparison_df`` (columns ``model``, ``mean_metric``,
        ``ci_low``, ``ci_high``). Sorting is handled internally.
    label_map:
        ``{model_identifier: pretty label}`` for full manual control of bar
        labels. Identifiers not present fall back to the raw model name.
    label_replacements / clean_labels / label_wrap:
        Further label tidying applied after ``label_map`` (see
        ``_format_comparison_labels``).
    xlim:
        Fixed x-axis limits shared by every stage. If None, computed from the CI
        extents (plus padding and room for value annotations).
    stage_titles:
        Optional per-stage titles (length = number of stages). Overrides
        ``title`` for the stages it covers.
    chance_line:
        If set (e.g. 0.5), draw a vertical reference line at this value.

    Returns
    -------
    dict with:
        "figs"        : list of matplotlib Figures, one per stage
        "plot_df"     : the sorted dataframe with the resolved "_label" column
        "saved_paths" : {stage_number: [paths]} (empty unless save=True)
    """
    df = comparison_df.copy()
    df = df.dropna(subset=[mean_col]).reset_index(drop=True)
    df = df.sort_values(mean_col, ascending=True).reset_index(drop=True)

    n_models = len(df)
    if n_models == 0:
        raise ValueError("comparison_df has no rows to plot.")

    labels = _format_comparison_labels(
        df[model_col].tolist(),
        label_map=label_map,
        label_replacements=label_replacements,
        clean_labels=clean_labels,
        label_wrap=label_wrap,
    )
    df["_label"] = labels

    means = df[mean_col].to_numpy(dtype=float)
    lows = df[ci_low_col].to_numpy(dtype=float) if ci_low_col in df else means
    highs = df[ci_high_col].to_numpy(dtype=float) if ci_high_col in df else means
    y_pos = np.arange(n_models)

    # Asymmetric error magnitudes (clipped to be non-negative).
    err_low = np.clip(means - lows, 0, None)
    err_high = np.clip(highs - means, 0, None)

    # Shared x-limits so frames don't jump.
    if xlim is None:
        lo = float(np.min(lows))
        hi = float(np.max(highs))
        span = max(hi - lo, 1e-6)
        x_lo = max(0.0, lo - span * 0.12)
        # leave headroom on the right for value annotations
        x_hi = hi + span * (0.28 if show_values else 0.12)
        if chance_line is not None:
            x_lo = min(x_lo, chance_line - span * 0.05)
            x_hi = max(x_hi, chance_line + span * 0.05)
        xlim = (x_lo, x_hi)

    if n_stages is None:
        n_stages = n_models
    n_stages = int(max(1, min(n_stages, n_models)))

    figs: List[Any] = []
    saved_paths: Dict[int, List[str]] = {}

    for stage in range(1, n_stages + 1):
        fig, ax = plt.subplots(figsize=figsize)

        revealed = np.arange(stage)
        colors = [base_color] * (stage - 1) + [highlight_color]

        ax.barh(
            y_pos[revealed],
            means[revealed],
            height=bar_height,
            color=colors,
            edgecolor=bar_edgecolor,
            alpha=bar_alpha,
            zorder=2,
        )

        if show_ci:
            ax.errorbar(
                means[revealed],
                y_pos[revealed],
                xerr=np.vstack([err_low[revealed], err_high[revealed]]),
                fmt="none",
                ecolor=error_color,
                elinewidth=error_linewidth,
                capsize=capsize,
                zorder=3,
            )

        if chance_line is not None:
            ax.axvline(
                chance_line,
                color=chance_color,
                linestyle=chance_linestyle,
                linewidth=1.3,
                zorder=1,
            )
            if chance_label:
                if chance_label_pos == "top":
                    y_anno, va_anno = n_models - 0.55, "top"
                else:  # "bottom" (default) — sits by the x-axis, clear of the title
                    y_anno, va_anno = -0.45, "bottom"
                ax.text(
                    chance_line,
                    y_anno,
                    " " + chance_label,
                    color=chance_color,
                    va=va_anno,
                    ha="left",
                    fontsize=tick_fontsize,
                )

        # Fixed geometry across stages.
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, n_models - 0.5)
        ax.set_yticks(y_pos)

        if show_all_labels:
            ax.set_yticklabels(df["_label"].tolist(), fontsize=tick_fontsize)
            for i, tick in enumerate(ax.get_yticklabels()):
                tick.set_color(
                    revealed_label_color if i < stage else pending_label_color
                )
        else:
            tick_labels = [
                lbl if i < stage else "" for i, lbl in enumerate(df["_label"].tolist())
            ]
            ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)
            for tick in ax.get_yticklabels():
                tick.set_color(revealed_label_color)

        ax.tick_params(axis="x", labelsize=tick_fontsize)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

        stage_title = title
        if stage_titles is not None and stage - 1 < len(stage_titles):
            stage_title = stage_titles[stage - 1]
        if stage_title is not None:
            ax.set_title(stage_title, fontsize=title_fontsize)

        if grid:
            ax.grid(axis=grid_axis, alpha=grid_alpha, zorder=0)
            ax.set_axisbelow(True)

        if show_values:
            for i in revealed:
                x_anno = (highs[i] if show_ci else means[i]) + value_pad
                text = value_fmt.format(means[i])
                if annotate_ci:
                    text += " " + ci_fmt.format(lows[i], highs[i])
                ax.text(
                    x_anno,
                    y_pos[i],
                    text,
                    va="center",
                    ha="left",
                    fontsize=value_fontsize,
                )

        fig.tight_layout()

        if save:
            stage_fname = (
                f"{filename}_{regime}_stage{stage}" if regime else f"{filename}_stage{stage}"
            )
            stage_paths = save_plot(
                fig=fig,
                rel_dir=rel_dir,
                filename=stage_fname,
                dpi=dpi,
                paper_dirs=paper_dirs,
                close=False,
            )
            saved_paths[stage] = stage_paths

        if close:
            plt.close(fig)
        else:
            figs.append(fig)

    return {
        "figs": figs,
        "plot_df": df,
        "saved_paths": saved_paths,
    }


def build_coef_comparison_table(
    coef_by_model: Mapping[str, pd.DataFrame],
    *,
    model_label_map: Optional[Mapping[str, str]] = None,
    feature_label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = True,
    decimals: Optional[int] = 4,
) -> pd.DataFrame:
    """
    Stack per-model coefficient summaries into one tidy comparison table.

    Each input value is a ``get_coef_summary`` DataFrame; empty entries (e.g. a
    dummy baseline) are skipped. The output has one row per (model, feature)
    with a ``model`` display label, the raw ``feature`` name and a cleaned
    ``feature_label``, followed by the original coefficient columns (coef,
    odds_ratio, CIs, sig_ci, ...). Numeric columns are rounded to ``decimals``.
    """
    frames: List[pd.DataFrame] = []
    for name, coef in coef_by_model.items():
        if coef is None or len(coef) == 0:
            continue
        d = coef.copy()
        model_label = (
            model_label_map[name] if (model_label_map and name in model_label_map) else name
        )
        d.insert(0, "model", model_label)
        d.insert(1, "model_key", name)
        d["feature_label"] = _format_comparison_labels(
            d["feature"].tolist(),
            label_map=feature_label_map,
            label_replacements=label_replacements,
            clean_labels=clean_labels,
        )
        frames.append(d)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    front = ["model", "model_key", "feature", "feature_label"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    if decimals is not None:
        num_cols = out.select_dtypes("number").columns
        out[num_cols] = out[num_cols].round(decimals)

    return out

