# answer_loc_viz.py
import numpy as np
import pandas as pd
from typing import Iterable, Mapping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.predictive_modeling.answer_loc.answer_loc_eval import ModelEvaluationResult


def show_model_results(
    results: Mapping[str, ModelEvaluationResult],
    labels: Iterable[int] = (0, 1, 2, 3),
) -> None:
    for model_name, res in results.items():
        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        acc = res.accuracy
        n = res.n_test
        y_true = np.asarray(res.y_true)
        y_pred = np.asarray(res.y_pred)

        print(f"Number of test trials: {n}")
        print(f"Accuracy: {acc:.3f}\n")

        cm = confusion_matrix(y_true, y_pred, labels=list(labels))
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels],
            columns=[f"pred_{l}" for l in labels],
        )
        print("Confusion Matrix:")
        print(cm_df.to_string())
        print()

        true_counts = (
            pd.Series(y_true)
            .value_counts()
            .sort_index()
            .rename("count")
        )
        print("True label distribution (label → count):")
        print(true_counts.to_string())
        print()

        pred_counts = (
            pd.Series(y_pred)
            .value_counts()
            .sort_index()
            .rename("count")
        )
        print("Prediction distribution (label → count):")
        print(pred_counts.to_string())
        print()


def plot_confusion_heatmap(
    y_true,
    y_pred,
    labels=(0, 1, 2, 3),
    include_minus1=False,
    normalize=False,
    title="Confusion matrix",
) -> None:
    if include_minus1:
        labels = (-1,) + tuple(labels)

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    index_names = [f"true_{l}" for l in labels]
    col_names   = [f"pred_{l}" for l in labels]
    cm_df = pd.DataFrame(cm, index=index_names, columns=col_names)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.show()
