# viz_utils.py

from typing import Optional, Iterable, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.plot_output import save_output

# NOTE: ``maybe_save_plot`` lived here and was the third plot-saving path in the
# project (``docs/restructure-map.md`` §1.2). It is gone: ``save_output`` already
# takes ``save`` and returns an empty result when it is False, so the wrapper had
# nothing left to do. Removed 2026-09-20 with T1.3.


def plot_confusion_heatmap(
    y_true,
    y_pred,
    labels: Iterable,
    title: str = "Confusion matrix",
    *,
    normalize: bool = False,
    save: Optional[bool] = None,
    to_paper=None,
    subdir: Optional[str] = None,
    plot: str = "confusion_matrix",
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """
    Confusion matrix heatmap with optional saving.

    """

    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    index_names = [f"true_{l}" for l in labels]
    col_names = [f"pred_{l}" for l in labels]
    cm_df = pd.DataFrame(cm, index=index_names, columns=col_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        ax=ax,
    )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    plt.tight_layout()

    saved_paths = save_output(
        fig,
        analysis="correctness_prediction",
        plot=plot,
        tables={"matrix": cm_df.reset_index(names="row")},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
        scale="normalized" if normalize else "raw",
        **facets,
    ).paths

    return fig, cm_df, saved_paths

