# src/predictive_modeling/answer_correctness/answer_correctness_viz.py

import numpy as np
import pandas as pd
from typing import Iterable, Mapping
from sklearn.metrics import confusion_matrix

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

    This mirrors answer_loc.show_model_results but for a binary target.
    """
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
        print(f"Negative (incorrect) trials: {res.n_negative}\n")

        cm = confusion_matrix(y_true, y_pred, labels=list(labels))
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels],
            columns=[f"pred_{l}" for l in labels],
        )
        print("Confusion Matrix:")
        print(cm_df.to_string())
        print()
