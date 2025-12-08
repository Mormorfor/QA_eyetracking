# viz_utils.py

from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_heatmap(
    y_true,
    y_pred,
    labels: Iterable,
    include_minus1: bool = False,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> None:
    if include_minus1 and (-1 not in labels):
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
