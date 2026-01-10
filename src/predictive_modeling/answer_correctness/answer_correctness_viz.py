# src/predictive_modeling/answer_correctness/answer_correctness_viz.py

import numpy as np
import pandas as pd
from typing import Iterable, Mapping
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
