"""Model-result tables and the one-call display helper.

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
