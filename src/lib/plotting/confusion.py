"""Confusion-matrix heatmap.

Moved from `predictive_modeling/common/viz_utils.py` in stage D step 7. It is in
`lib/plotting/` by the test `restructure-map.md` section 4 sets: *if it mentions
an eye-tracking column name, it is not lib*. This takes `y_true`, `y_pred` and
`labels` and names no project vocabulary at all, so it qualifies -- and its one
caller (`run_model_bundles`) becomes an analysis in stage E, which would
otherwise have left a parked exploration importing from `analyses/`.

It briefly sat in `viz/` because it calls `save_output`, which was still there;
`plot_output.py` moved to `lib/plotting/output.py` in the same step, so the two
now sit together as the map's section 3 tree intends.

`maybe_save_plot` used to live beside it and was the third plot-saving path in
the project (`restructure-map.md` section 1.2). It went with T1.3 on 2026-09-20:
`save_output` already takes `save` and returns an empty result when it is False,
so the wrapper had nothing left to do.
"""

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.lib.plotting.output import save_output


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

