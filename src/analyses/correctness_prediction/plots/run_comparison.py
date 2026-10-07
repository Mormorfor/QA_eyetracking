"""Comparing saved runs against each other, including the staged presentation chart.

`collect_correctness_run_reports` **scans a directory** rather than taking a curated
list, so anything ever saved under that folder joins the comparison figure. Check the
folder before generating a final figure.

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


def _infer_model_family(model_name: str) -> Optional[str]:
    if not isinstance(model_name, str):
        return None

    name = model_name.lower()

    if "log_reg" in name or "logreg" in name:
        return "logreg"
    if "glmer" in name and "julia" not in name:
        return "glmer"
    if "julia" in name:
        return "julia"

    return None


def collect_correctness_run_reports(
    report_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    filename: str = "model_summary*__summary.csv",
    recursive: bool = True,
    sort_by: str = "balanced_accuracy",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Collect correctness run summary CSVs from one or more directories.

    Parameters
    ----------
    report_dirs:
        One folder or a list of folders that contain run report CSVs.
        Example:
            "reports/report_data/answer_correctness/logreg"
            [
                "reports/report_data/answer_correctness/logreg",
                "reports/report_data/answer_correctness/glmer",
                "reports/report_data/answer_correctness/julia",
            ]

    filename:
        Glob for the run-summary CSVs. Default "model_summary*__summary.csv",
        which is what `save_output` writes: the stem is `model_summary` plus the
        run/model facets, and `__summary` is the `tables=` key. The pre-T1.3
        layout wrote a bare `model_summary.csv` one directory per run, so a
        plain filename no longer matches anything.

    recursive:
        If True, searches all nested subfolders.

    sort_by:
        Column to sort the final table by.

    ascending:
        Sort direction.

    Returns
    -------
    pd.DataFrame
        Combined dataframe of all found run summaries.
    """
    if isinstance(report_dirs, (str, Path)):
        report_dirs = [report_dirs]

    csv_paths: List[Path] = []

    for folder in report_dirs:
        folder = Path(folder)
        if recursive:
            csv_paths.extend(folder.rglob(filename))
        else:
            csv_paths.extend(folder.glob(filename))

    frames = []
    for csv_path in sorted(set(csv_paths)):

        df = pd.read_csv(csv_path)
        df["source_csv"] = str(csv_path)
        df["source_folder"] = str(csv_path.parent)

        parts = list(csv_path.parts)
        df["model_family"] = df["model"].apply(_infer_model_family)
        frames.append(df)


    out = pd.concat(frames, ignore_index=True)

    preferred_cols = [
        "run_identifier",
        "model_family",
        "model",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "n_test",
        "n_features",
        "trained_feature_cols",
        "source_folder",
    ]
    existing_preferred = [c for c in preferred_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_preferred]
    out = out[existing_preferred + remaining]

    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    return out


def plot_correctness_run_comparison(
    summary_df: pd.DataFrame,
    metric_col: str = "balanced_accuracy",
    label_col: Optional[str] = None,
    top_n: Optional[int] = None,
    figsize: tuple = (12, 8),
    title: Optional[str] = None,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "run_comparison",
    filename: str = "run_comparison_balanced_accuracy",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    ytick_fontsize: Optional[float] = None,
    label_wrap: Optional[int] = None,
    label_split_on_sep: bool = False,
    label_fields: Sequence[str] = ("run_identifier", "model_family", "n_features"),
    clean_labels: bool = True,
    label_replacements: Optional[Mapping[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    value_fmt: str = "{:.3f}",
    show_values: bool = True,
    **facets,
):
    """
    Create a horizontal bar plot comparing runs by balanced accuracy.

    Parameters
    ----------
    summary_df:
        Combined dataframe returned by collect_correctness_run_reports().

    metric_col:
        Metric to plot. Default: balanced_accuracy

    label_col:
        Column to use as bar labels.
        If None, a readable label is constructed automatically.

    top_n:
        Optionally only plot the top N runs.

    ytick_fontsize:
        Font size for the y-axis (run label) tick text. If None, uses the
        matplotlib default.

    label_wrap:
        If set, wrap each y-axis label to this maximum character width,
        breaking onto multiple lines.

    label_split_on_sep:
        If True, put each "|"-separated part of the label on its own line.
        Takes precedence over label_wrap when the label contains "|".

    label_fields:
        Which fields (and in what order) make up the auto-generated label
        when label_col is None. Choose from "run_identifier", "model_family",
        "n_features". E.g. ("run_identifier", "n_features") drops the model
        family (the "logreg" part).

    clean_labels:
        If True (default), tidy y-axis labels for presentation: apply
        ``label_replacements`` then replace remaining underscores with spaces
        and collapse repeated whitespace.

    label_replacements:
        Optional {find: replace} mapping applied to each label before the
        underscore cleanup. Use it for full control over wording, e.g.
        {"correct+mean_wrong RT": "Correct vs. mean-wrong RT", "RT": "reaction time"}.

    xlabel / ylabel:
        Axis-label overrides for presentation. If None, the x-axis is derived
        from ``metric_col`` (underscores -> spaces, title-cased) and the y-axis
        is left blank (the per-bar labels are self-explanatory). Pass an empty
        string to force a blank axis label.

    value_fmt:
        Format string for the per-bar value annotations. Default "{:.3f}".

    show_values:
        If True (default), annotate each bar with its metric value.

    Returns
    -------
    fig, plot_df, saved_paths
    """

    df = summary_df.copy()

    if label_col is None:
        def make_label(row):
            field_parts = {
                "run_identifier": (
                    row["run_identifier"]
                    if "run_identifier" in row and pd.notna(row["run_identifier"]) and str(row["run_identifier"]).strip()
                    else None
                ),
                "model_family": (
                    row["model_family"]
                    if "model_family" in row and pd.notna(row["model_family"])
                    else None
                ),
                "n_features": (
                    str(row["n_features"]) + " features"
                    if "n_features" in row and pd.notna(row["n_features"])
                    else None
                ),
            }

            parts = [field_parts[f] for f in label_fields if field_parts.get(f)]
            return " | ".join(parts)

        df["_plot_label"] = df.apply(make_label, axis=1)
        label_col = "_plot_label"

    df = df.dropna(subset=[metric_col]).copy()
    df = df.sort_values(metric_col, ascending=False)

    if top_n is not None:
        df = df.head(top_n).copy()

    df = df.sort_values(metric_col, ascending=True)

    import re

    def clean_label(text: str) -> str:
        text = str(text)
        if label_replacements:
            for find, replace in label_replacements.items():
                text = text.replace(find, replace)
        if clean_labels:
            text = text.replace("_", " ")
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def wrap_label(text: str) -> str:
        text = clean_label(text)
        if label_split_on_sep and "|" in text:
            return "\n".join(part.strip() for part in text.split("|"))
        if label_wrap is not None:
            import textwrap
            return "\n".join(textwrap.wrap(text, width=label_wrap)) or text
        return text

    df = df.copy()
    df[label_col] = df[label_col].map(wrap_label)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df[label_col].astype(str), df[metric_col])
    ax.set_xlabel(xlabel if xlabel is not None else metric_col.replace("_", " ").title())
    ax.set_ylabel(ylabel if ylabel is not None else "")
    ax.set_title(title or f"Comparison of runs by {metric_col.replace('_', ' ')}")

    if ytick_fontsize is not None:
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

    if show_values:
        # Extend the x-axis so the value annotations don't spill past the frame.
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.08)
        for i, val in enumerate(df[metric_col]):
            ax.text(val, i, " " + value_fmt.format(val), va="center")

    plt.tight_layout()

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


# ---------------------------------------------------------------------
# Staged cross-validation model comparison (presentation bar chart)
# ---------------------------------------------------------------------

def _format_comparison_labels(
    models: Sequence[str],
    *,
    label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = True,
    label_wrap: Optional[int] = None,
) -> List[str]:
    """Turn raw model identifiers into presentation-ready bar labels.

    Order of operations per label: explicit ``label_map`` lookup (falls back to
    the raw identifier) -> ``label_replacements`` substring swaps -> underscore
    cleanup (if ``clean_labels``) -> optional wrapping to ``label_wrap`` chars.
    """
    import re
    import textwrap

    out: List[str] = []
    for model in models:
        text = str(label_map[model]) if (label_map and model in label_map) else str(model)
        if label_replacements:
            for find, replace in label_replacements.items():
                text = text.replace(find, replace)
        if clean_labels:
            text = text.replace("_", " ")
            text = re.sub(r"\s+", " ", text).strip()
        if label_wrap is not None:
            text = "\n".join(textwrap.wrap(text, width=label_wrap)) or text
        out.append(text)
    return out


def plot_cv_model_comparison_staged(
    comparison_df: pd.DataFrame,
    *,
    model_col: str = "model",
    mean_col: str = "mean_metric",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    swap_models: Optional[Sequence[Tuple[str, str]]] = None,
    # ---- labels --------------------------------------------------------
    label_map: Optional[Mapping[str, str]] = None,
    label_replacements: Optional[Mapping[str, str]] = None,
    clean_labels: bool = True,
    label_wrap: Optional[int] = None,
    # ---- figure / axes -------------------------------------------------
    figsize: Tuple[float, float] = (9.0, 6.0),
    xlim: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = "Balanced accuracy",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    stage_titles: Optional[Sequence[str]] = None,
    # ---- fonts ---------------------------------------------------------
    title_fontsize: float = 16,
    xlabel_fontsize: float = 14,
    ylabel_fontsize: float = 14,
    tick_fontsize: float = 12,
    value_fontsize: float = 12,
    # ---- bars ----------------------------------------------------------
    base_color: str = "#4C72B0",
    highlight_color: str = "#DD8452",
    bar_height: float = 0.62,
    bar_edgecolor: str = "none",
    bar_alpha: float = 1.0,
    # ---- error bars (CIs) ---------------------------------------------
    show_ci: bool = True,
    capsize: float = 4,
    error_color: str = "#333333",
    error_linewidth: float = 1.4,
    # ---- value annotations --------------------------------------------
    show_values: bool = True,
    value_fmt: str = "{:.3f}",
    value_pad: float = 0.006,
    annotate_ci: bool = False,
    ci_fmt: str = "[{:.3f}, {:.3f}]",
    # ---- pending (not-yet-revealed) styling ---------------------------
    show_all_labels: bool = True,
    revealed_label_color: str = "#000000",
    pending_label_color: str = "#BBBBBB",
    # ---- reference (chance) line --------------------------------------
    chance_line: Optional[float] = None,
    chance_label: Optional[str] = None,
    chance_label_pos: str = "bottom",
    chance_color: str = "#888888",
    chance_linestyle: str = "--",
    # ---- grid ----------------------------------------------------------
    grid: bool = True,
    grid_axis: str = "x",
    grid_alpha: float = 0.4,
    # ---- staging / saving ---------------------------------------------
    n_stages: Optional[int] = None,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "cross_val_comp",
    filename: str = "cross_val_comparison",
    regime: Optional[str] = None,
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
) -> Dict[str, Any]:
    """
    Build a horizontal bar chart comparing models by a cross-validated metric
    (with confidence intervals), revealed in stages.

    The models are sorted low -> high and one bar is revealed per stage, from
    lowest to highest. The newly added bar is drawn in ``highlight_color`` while
    previously revealed bars use ``base_color``. Every stage uses identical
    figure size, axis limits and y-tick labels (all category labels are always
    shown when ``show_all_labels``), so the saved frames do not jump between
    stages -- suitable for an animated/click-through conference slide.

    Parameters
    ----------
    comparison_df:
        One row per model with the mean metric and CI bounds, e.g. the output of
        ``build_cv_model_comparison_df`` (columns ``model``, ``mean_metric``,
        ``ci_low``, ``ci_high``). Sorting is handled internally.
    label_map:
        ``{model_identifier: pretty label}`` for full manual control of bar
        labels. Identifiers not present fall back to the raw model name.
    label_replacements / clean_labels / label_wrap:
        Further label tidying applied after ``label_map`` (see
        ``_format_comparison_labels``).
    xlim:
        Fixed x-axis limits shared by every stage. If None, computed from the CI
        extents (plus padding and room for value annotations).
    stage_titles:
        Optional per-stage titles (length = number of stages). Overrides
        ``title`` for the stages it covers.
    chance_line:
        If set (e.g. 0.5), draw a vertical reference line at this value.

    Returns
    -------
    dict with:
        "figs"        : list of matplotlib Figures, one per stage
        "plot_df"     : the sorted dataframe with the resolved "_label" column
        "saved_paths" : {stage_number: [paths]} (empty unless save=True)
    """
    df = comparison_df.copy()
    df = df.dropna(subset=[mean_col]).reset_index(drop=True)
    df = df.sort_values(mean_col, ascending=True).reset_index(drop=True)

    # Optional manual reordering: swap the positions of given model pairs after
    # the ascending sort (e.g. to switch two near-tied bars). Uses model keys.
    if swap_models:
        order = df[model_col].tolist()
        for a, b in swap_models:
            missing = [m for m in (a, b) if m not in order]
            if missing:
                raise ValueError(f"swap_models: model(s) not found: {missing}")
            ia, ib = order.index(a), order.index(b)
            order[ia], order[ib] = order[ib], order[ia]
        df = (
            df.set_index(model_col)
            .loc[order]
            .reset_index()
        )

    n_models = len(df)
    if n_models == 0:
        raise ValueError("comparison_df has no rows to plot.")

    labels = _format_comparison_labels(
        df[model_col].tolist(),
        label_map=label_map,
        label_replacements=label_replacements,
        clean_labels=clean_labels,
        label_wrap=label_wrap,
    )
    df["_label"] = labels

    means = df[mean_col].to_numpy(dtype=float)
    lows = df[ci_low_col].to_numpy(dtype=float) if ci_low_col in df else means
    highs = df[ci_high_col].to_numpy(dtype=float) if ci_high_col in df else means
    y_pos = np.arange(n_models)

    # Asymmetric error magnitudes (clipped to be non-negative).
    err_low = np.clip(means - lows, 0, None)
    err_high = np.clip(highs - means, 0, None)

    # Shared x-limits so frames don't jump.
    if xlim is None:
        lo = float(np.min(lows))
        hi = float(np.max(highs))
        span = max(hi - lo, 1e-6)
        x_lo = max(0.0, lo - span * 0.12)
        # leave headroom on the right for value annotations
        x_hi = hi + span * (0.28 if show_values else 0.12)
        if chance_line is not None:
            x_lo = min(x_lo, chance_line - span * 0.05)
            x_hi = max(x_hi, chance_line + span * 0.05)
        xlim = (x_lo, x_hi)

    if n_stages is None:
        n_stages = n_models
    n_stages = int(max(1, min(n_stages, n_models)))

    figs: List[Any] = []
    saved_paths: Dict[int, List[str]] = {}

    for stage in range(1, n_stages + 1):
        fig, ax = plt.subplots(figsize=figsize)

        revealed = np.arange(stage)
        colors = [base_color] * (stage - 1) + [highlight_color]

        ax.barh(
            y_pos[revealed],
            means[revealed],
            height=bar_height,
            color=colors,
            edgecolor=bar_edgecolor,
            alpha=bar_alpha,
            zorder=2,
        )

        if show_ci:
            ax.errorbar(
                means[revealed],
                y_pos[revealed],
                xerr=np.vstack([err_low[revealed], err_high[revealed]]),
                fmt="none",
                ecolor=error_color,
                elinewidth=error_linewidth,
                capsize=capsize,
                zorder=3,
            )

        if chance_line is not None:
            ax.axvline(
                chance_line,
                color=chance_color,
                linestyle=chance_linestyle,
                linewidth=1.3,
                zorder=1,
            )
            if chance_label:
                if chance_label_pos == "top":
                    y_anno, va_anno = n_models - 0.55, "top"
                else:  # "bottom" (default) — sits by the x-axis, clear of the title
                    y_anno, va_anno = -0.45, "bottom"
                ax.text(
                    chance_line,
                    y_anno,
                    " " + chance_label,
                    color=chance_color,
                    va=va_anno,
                    ha="left",
                    fontsize=tick_fontsize,
                )

        # Fixed geometry across stages.
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, n_models - 0.5)
        ax.set_yticks(y_pos)

        if show_all_labels:
            ax.set_yticklabels(df["_label"].tolist(), fontsize=tick_fontsize)
            for i, tick in enumerate(ax.get_yticklabels()):
                tick.set_color(
                    revealed_label_color if i < stage else pending_label_color
                )
        else:
            tick_labels = [
                lbl if i < stage else "" for i, lbl in enumerate(df["_label"].tolist())
            ]
            ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)
            for tick in ax.get_yticklabels():
                tick.set_color(revealed_label_color)

        ax.tick_params(axis="x", labelsize=tick_fontsize)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

        stage_title = title
        if stage_titles is not None and stage - 1 < len(stage_titles):
            stage_title = stage_titles[stage - 1]
        if stage_title is not None:
            ax.set_title(stage_title, fontsize=title_fontsize)

        if grid:
            ax.grid(axis=grid_axis, alpha=grid_alpha, zorder=0)
            ax.set_axisbelow(True)

        if show_values:
            for i in revealed:
                x_anno = (highs[i] if show_ci else means[i]) + value_pad
                text = value_fmt.format(means[i])
                if annotate_ci:
                    text += " " + ci_fmt.format(lows[i], highs[i])
                ax.text(
                    x_anno,
                    y_pos[i],
                    text,
                    va="center",
                    ha="left",
                    fontsize=value_fontsize,
                )

        fig.tight_layout()

        stage_paths = save_output(
            fig,
            analysis="correctness_prediction",
            plot=plot,
            tables={"data": df},
            save=save,
            to_paper=to_paper,
            dpi=dpi,
            close=False,
            subdir=subdir,
            regime=regime,
            stage=stage,
            **facets,
        ).paths
        if stage_paths:
            saved_paths[stage] = stage_paths

        if close:
            plt.close(fig)
        else:
            figs.append(fig)

    return {
        "figs": figs,
        "plot_df": df,
        "saved_paths": saved_paths,
    }
