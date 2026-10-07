"""Coefficient figures and the across-run coefficient comparison table.

The CIs behind these come from `modeling/inference.py`; `collect_logreg_coef_summaries`
defaults to the participant-clustered bootstrap for the paper path (T3.3).

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
from src.analyses.correctness_prediction.plots.run_comparison import _format_comparison_labels



def plot_coef_summary_barh(
    coef_summary: pd.DataFrame,
    value_col: Optional[str] = "coef",
    top_k: int = 100,
    title: Optional[str] = None,
    model_name: Optional[str] = None,
    h_or_g: Optional[str] = "all_participants",
    figsize: Optional[Tuple[int, int]] = None,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "coefficients",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    significant_only: bool = True,
    significance_eps: float = 0.0,
    # ---- presentation touch-ups (all optional, defaults preserve behavior) ----
    feature_label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = False,
    label_wrap: Optional[int] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_color: Optional[str] = None,
    bar_thickness: Optional[float] = None,
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    **facets,
):
    """
    Horizontal bar plot of top coefficients (by absolute magnitude), overlay 95% CI error bars.
    Can exclude insignificant.

    Presentation touch-ups (optional, off by default):
        feature_label_map / label_replacements / clean_labels / label_wrap
            tidy the y-axis feature names (see ``_format_comparison_labels``).
        xlabel / ylabel
            axis-label overrides (default: ``value_col`` / "feature").
        bar_color
            single bar color.
        title_fontsize / label_fontsize / tick_fontsize
            font sizes for the title, axis labels and tick labels.
    """

    df = coef_summary.copy()

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if "abs_coef" not in df.columns:
        df["abs_coef"] = df[value_col].abs()
    else:
        df["abs_coef"] = pd.to_numeric(df["abs_coef"], errors="coerce")

    if "ci_low" not in df.columns:
        df["ci_low"] = np.nan
    else:
        df["ci_low"] = pd.to_numeric(df["ci_low"], errors="coerce")

    if "ci_high" not in df.columns:
        df["ci_high"] = np.nan
    else:
        df["ci_high"] = pd.to_numeric(df["ci_high"], errors="coerce")

    if "sig_ci" not in df.columns:
        df["sig_ci"] = False

    sig_mask = (df["ci_low"] > significance_eps) | (df["ci_high"] < -significance_eps)
    df["significant"] = sig_mask

    if significant_only:
        df = df[df["significant"]].copy()

    df = df.sort_values("abs_coef", ascending=False).head(int(top_k)).copy()
    df = df.sort_values(value_col, ascending=True)

    if figsize is not None:
        width, height = figsize
    else:
        # Height scales with the number of bars; few-feature models stay short
        # instead of becoming a single fat block in a tall frame.
        n_bars = len(df)
        row_height = 0.55
        min_height = 1.8
        max_height = 30
        width = 9
        height = min(max(min_height, n_bars * row_height + 1.0), max_height)

    fig, ax = plt.subplots(figsize=(width, height))

    y = np.arange(len(df))
    x = df[value_col].to_numpy()

    barh_kwargs = {}
    if bar_color is not None:
        barh_kwargs["color"] = bar_color
    if bar_thickness is not None:
        barh_kwargs["height"] = bar_thickness

    ax.barh(y, x, **barh_kwargs)
    # Keep a consistent margin around the bars so a single bar isn't flush to
    # the frame edges.
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.axvline(0, linewidth=1)

    display_labels = _format_comparison_labels(
        df["feature"].tolist(),
        label_map=feature_label_map,
        label_replacements=label_replacements,
        clean_labels=clean_labels,
        label_wrap=label_wrap,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels)

    ax.set_xlabel(xlabel if xlabel is not None else value_col,
                  fontsize=label_fontsize)
    ax.set_ylabel(ylabel if ylabel is not None else "feature",
                  fontsize=label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    lo = df["ci_low"].to_numpy()
    hi = df["ci_high"].to_numpy()

    mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(x)
    if mask.any():
        xerr = np.vstack([x[mask] - lo[mask], hi[mask] - x[mask]])
        ax.errorbar(
            x[mask],
            y[mask],
            xerr=xerr,
            fmt="none",
            capsize=2,
            linewidth=1,
            color="black",
            ecolor="black",
        )

    plt.tight_layout()

    if filename is None:
        mn = model_name or "model"
        hg = h_or_g or "group"
        suffix = "_sigonly" if significant_only else ""
        filename = f"{mn}_{hg}_top{top_k}_{value_col}{suffix}"

    saved_paths = save_output(
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
    return fig, df, saved_paths


def build_coef_comparison_table(
    coef_by_model: Mapping[str, pd.DataFrame],
    *,
    model_label_map: Optional[Mapping[str, str]] = None,
    feature_label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = True,
    decimals: Optional[int] = 4,
) -> pd.DataFrame:
    """
    Stack per-model coefficient summaries into one tidy comparison table.

    Each input value is a ``get_coef_summary`` DataFrame; empty entries (e.g. a
    dummy baseline) are skipped. The output has one row per (model, feature)
    with a ``model`` display label, the raw ``feature`` name and a cleaned
    ``feature_label``, followed by the original coefficient columns (coef,
    odds_ratio, CIs, sig_ci, ...). Numeric columns are rounded to ``decimals``.
    """
    frames: List[pd.DataFrame] = []
    for name, coef in coef_by_model.items():
        if coef is None or len(coef) == 0:
            continue
        d = coef.copy()
        model_label = (
            model_label_map[name] if (model_label_map and name in model_label_map) else name
        )
        d.insert(0, "model", model_label)
        d.insert(1, "model_key", name)
        d["feature_label"] = _format_comparison_labels(
            d["feature"].tolist(),
            label_map=feature_label_map,
            label_replacements=label_replacements,
            clean_labels=clean_labels,
        )
        frames.append(d)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    front = ["model", "model_key", "feature", "feature_label"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    if decimals is not None:
        num_cols = out.select_dtypes("number").columns
        out[num_cols] = out[num_cols].round(decimals)

    return out
