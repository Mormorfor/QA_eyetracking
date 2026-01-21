# src/predictive_modeling/answer_correctness/answer_correctness_viz.py

from __future__ import annotations
from typing import Optional, Tuple
from typing import Iterable, Mapping, Any, Dict

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.predictive_modeling.answer_correctness.answer_correctness_eval import (
    CorrectnessEvaluationResult,
)


def show_correctness_model_results(
    results: Mapping[str, CorrectnessEvaluationResult],
    labels: Iterable[int] = (0, 1),
) -> None:
    """
    Print per-model summary statistics and a confusion matrix (text form)
    for correctness prediction (is_correct = 0/1).
    Also prints precision/recall/F1.
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
            f"\n  macro   P/R/F1: {macro_p:.3f} / {macro_r:.3f} / {macro_f1:.3f}"
            f"\n  weighted P/R/F1: {weighted_p:.3f} / {weighted_r:.3f} / {weighted_f1:.3f}"
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels],
            columns=[f"pred_{l}" for l in labels],
        )
        print("\nConfusion Matrix:")
        print(cm_df.to_string())
        print()



def plot_coef_summary_barh(
    coef_summary: pd.DataFrame,
    value_col: Optional[str] = 'standardized_coef',
    top_k: int = 75,
    title: Optional[str] = None,
    model_name: Optional[str] = None,
    h_or_g: Optional[str] = 'hunters',
    figsize: Tuple[int, int] = (9, 7),
    save: bool = False,
    output_dir: Optional[str] = '../reports/plots/answer_correctness_coefficients',
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Horizontal bar plot of top coefficients (by absolute magnitude).

    """

    df = coef_summary.copy()
    abs_col = "abs_coef"
    df = df.sort_values(abs_col, ascending=False).head(int(top_k)).copy()

    df = df.sort_values(value_col, ascending=True)

    n_bars = len(df)
    row_height = 0.30
    min_height = 6
    max_height = 30

    height = min(max(min_height, n_bars * row_height), max_height)

    fig, ax = plt.subplots(figsize=(figsize[0], height))
    ax.barh(df["feature"], df[value_col])
    ax.axvline(0, linewidth=1)  # reference line at 0

    ax.set_xlabel(value_col)
    ax.set_ylabel("feature")
    if title:
        ax.set_title(title)

    if "odds_ratio" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                or_val = float(row["odds_ratio"])
            except Exception:
                continue
            ax.text(
                x=row[value_col],
                y=i,
                s=f"  OR={or_val:.2f}",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()

    if save:
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            os.makedirs(output_dir, exist_ok=True)
            fname = f"{model_name}_{h_or_g}.png"
            outpath = os.path.join(output_dir, fname)
            fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if "__abs_tmp__" in df.columns:
        df = df.drop(columns=["__abs_tmp__"])

    return fig, df




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
        save_path: Optional[str] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
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

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, agg
