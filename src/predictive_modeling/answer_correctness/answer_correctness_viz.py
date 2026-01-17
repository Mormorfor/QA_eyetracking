# src/predictive_modeling/answer_correctness/answer_correctness_viz.py

from __future__ import annotations
from typing import Optional, Tuple
from typing import Iterable, Mapping

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

        # ---- precision / recall / f1 (per class + macro + weighted)
        # zero_division=0 avoids warnings when a class is never predicted.
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

        # ---- confusion matrix
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
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Horizontal bar plot of top coefficients (by absolute magnitude).

    """

    df = coef_summary.copy()

    abs_col = None
    if value_col == "standardized_coef" and "abs_standardized_coef" in df.columns:
        abs_col = "abs_standardized_coef"
    elif value_col == "coef" and "abs_coef" in df.columns:
        abs_col = "abs_coef"

    if abs_col is None:
        abs_col = "__abs_tmp__"
        df[abs_col] = df[value_col].abs()

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
        os.makedirs(output_dir, exist_ok=True)

        fname = (
            f"{model_name}_{h_or_g}.png"
        )
        outpath = os.path.join(output_dir, fname)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if "__abs_tmp__" in df.columns:
        df = df.drop(columns=["__abs_tmp__"])

    return fig, df