# viz_utils.py

from typing import Iterable, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.plot_output import save_plot

def maybe_save_plot(
    fig,
    save: bool,
    rel_dir: str,
    filename: str,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    ext: str = "png",
    close: bool = False,
):
    if not save:
        return []
    return save_plot(
        fig=fig,
        rel_dir=rel_dir,
        filename=filename,
        dpi=dpi,
        ext=ext,
        paper_dirs=paper_dirs,
        close=close,
    )


def plot_confusion_heatmap(
    y_true,
    y_pred,
    labels: Iterable,
    title: str = "Confusion matrix",
    *,
    normalize: bool = False,
    save: bool = False,
    rel_dir: str = "answer_correctness/confusion",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
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

    if filename is None:
        norm_tag = "normalized" if normalize else "raw"
        filename = f"confusion_matrix_{norm_tag}"

    saved_paths = maybe_save_plot(
        fig=fig,
        save=save,
        rel_dir=rel_dir,
        filename=filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return fig, cm_df, saved_paths