"""Random-effects figures for the mixed-model backends. **Parked.**

Split out of `answer_correctness_viz.py` in stage E. These read `get_random_effects()`
and `get_random_effect_variance_summary()`, which only the Julia backend implements
(`todo.md` T2.3) -- so they are unusable with the logistic regression the paper reports,
and they belong with the backends rather than with the paper figures.

**Not to be confused with `analyses/attention_allocation/stats.py`**, which also says
"mixed effects" and is paper code: that is a *difference test* between screen areas, this
is *fitting the outcome* with a mixed model. Diana, 2026-10-07.
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


def plot_random_effects_barh(
    random_effects_df: pd.DataFrame,
    id_col: str,
    effect_col: str = "random_intercept",
    title: Optional[str] = None,
    top_n: int = 30,
    sort_by_abs: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "random_effects_barh",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    df = random_effects_df.copy()
    df = df[[id_col, effect_col]].dropna()

    if sort_by_abs:
        df = df.assign(_abs=df[effect_col].abs()).sort_values("_abs", ascending=False)
    else:
        df = df.sort_values(effect_col, ascending=False)

    df = df.head(top_n).copy()
    df = df.sort_values(effect_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df[id_col].astype(str), df[effect_col])
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Random effect")
    ax.set_ylabel(id_col)
    ax.set_title(title or f"Random effects: {id_col}")

    fig.tight_layout()

    paths = save_output(
        fig,
        analysis="correctness_prediction",
        plot=plot,
        tables={"data": df},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
        **facets,
    ).paths

    return fig, df, paths


def plot_random_effects_distribution(
    random_effects_df: pd.DataFrame,
    effect_col: str = "random_intercept",
    title: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (8, 5),
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "random_effects_distribution",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    vals = pd.to_numeric(random_effects_df[effect_col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=bins)
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Random effect")
    ax.set_ylabel("Count")
    ax.set_title(title or "Random-effects distribution")

    fig.tight_layout()

    paths = save_output(
        fig,
        analysis="correctness_prediction",
        plot=plot,
        tables={"data": vals},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
        **facets,
    ).paths

    return fig, vals, paths


def summarize_random_effects(
    random_effects_df: pd.DataFrame,
    group_name: str,
    effect_col: str = "random_intercept",
) -> pd.DataFrame:
    vals = pd.to_numeric(random_effects_df[effect_col], errors="coerce").dropna()

    return pd.DataFrame([{
        "group_name": group_name,
        "n_levels": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "min": float(vals.min()),
        "q25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "q75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
        "mean_abs": float(vals.abs().mean()),
        "max_abs": float(vals.abs().max()),
    }])
