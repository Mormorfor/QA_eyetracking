"""Collect saved cross-validation run reports and draw the model-comparison figure.

Note the figure's contents are whatever is on disk: these functions SCAN the
given report directories rather than taking a curated model list, so any run
ever saved there joins the comparison. Check the folder before generating a
final figure.
"""

from typing import Iterable, List, Optional, Sequence, Union, Dict, Any, Tuple, Mapping
from pathlib import Path

import pandas as pd

from predictive_modeling.answer_correctness.answer_correctness_viz import plot_correctness_run_comparison, \
    collect_correctness_run_reports
from src.viz.plot_output import save_output


def collect_and_plot_correctness_runs(
    report_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    filename: str = "model_summary.csv",
    recursive: bool = True,
    sort_by: str = "balanced_accuracy",
    ascending: bool = False,
    metric_col: str = "balanced_accuracy",
    label_col: Optional[str] = None,
    top_n: Optional[int] = None,
    figsize: tuple = (12, 8),
    title: Optional[str] = None,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "run_comparison_balanced_accuracy",
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
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    - collect all run summaries
    - plot balanced accuracy comparison
    - optionally save combined table and figure
    """
    summary_df = collect_correctness_run_reports(
        report_dirs=report_dirs,
        filename=filename,
        recursive=recursive,
        sort_by=sort_by,
        ascending=ascending,
    )

    fig, plot_df, plot_paths = plot_correctness_run_comparison(
        summary_df=summary_df,
        metric_col=metric_col,
        label_col=label_col,
        top_n=top_n,
        figsize=figsize,
        title=title,
        save=save,
        subdir=subdir,
        plot=plot,
        tables={"run_summaries": summary_df},
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        ytick_fontsize=ytick_fontsize,
        label_wrap=label_wrap,
        label_split_on_sep=label_split_on_sep,
        label_fields=label_fields,
        clean_labels=clean_labels,
        label_replacements=label_replacements,
        xlabel=xlabel,
        ylabel=ylabel,
        value_fmt=value_fmt,
        show_values=show_values,
    )

    return {
        "summary_df": summary_df,
        "plot_df": plot_df,
        "fig": fig,
        # The figure and its table are written by one call now, so there is a
        # single list of paths rather than two that could disagree.
        "saved_paths": plot_paths,
    }


def collect_correctness_runs_by_mode(
    base_dir: Union[str, Path],
    modes: Optional[Mapping[str, str]] = None,
    filename: str = "model_summary.csv",
    recursive: bool = True,
    metric_col: str = "balanced_accuracy",
    index_col: str = "run_identifier",
    decimals: Optional[int] = 4,
    sort_by_mode: Optional[str] = None,
    save_table: bool = False,
    subdir: Optional[str] = None,
    table_filename: str = "correctness_runs_by_mode",
    to_paper=None,
) -> Dict[str, Any]:
    """
    Build a comparison table of one metric (default: balanced accuracy) across
    the three testing modes.

    The resulting table has:
        - one column per testing mode (new_item, new_subject, both)
        - one row per run identifier (i.e. which features the model trained on)
        - cells holding the metric value for that run / mode combination

    Parameters
    ----------
    base_dir:
        Folder that contains the per-mode subfolders, e.g.
        "reports/report_data/answer_correctness".
    modes:
        Mapping of {column_label: subfolder_name}. Defaults to
        {"new_item": "new_item", "new_subject": "new_subject", "both": "both"}.
        The column order in the output follows this mapping's order.
    filename:
        Run summary CSV filename to look for. Default: "model_summary.csv".
    recursive:
        Search nested subfolders for the CSVs.
    metric_col:
        Column to place in the table cells. Default: "balanced_accuracy".
    index_col:
        Column used as the run identifier (table rows). Default: "run_identifier".
    decimals:
        If not None, round the metric cells to this many decimals.
    sort_by_mode:
        Optional mode column label to sort rows by (descending). If None, rows
        are sorted alphabetically by run identifier.
    save / subdir / plot / to_paper:
        One flag now covers the figure and its numbers -- they are written by the
        same save_output call, so a saved figure always has a saved table.

    Returns
    -------
    dict with:
        "pivot_df"  : wide table (index=run identifier, columns=modes)
        "long_df"   : tidy table with a "testing_mode" column
        "table_paths": list of saved CSV paths (empty unless save_table=True)
    """
    if modes is None:
        modes = {
            "new_item": "new_item",
            "new_subject": "new_subject",
            "both": "both",
        }

    base_dir = Path(base_dir)

    frames: List[pd.DataFrame] = []
    for mode_label, subfolder in modes.items():
        mode_dir = base_dir / subfolder
        if not mode_dir.exists():
            print(f"[collect_correctness_runs_by_mode] Skipping missing folder: {mode_dir}")
            continue

        mode_df = collect_correctness_run_reports(
            report_dirs=mode_dir,
            filename=filename,
            recursive=recursive,
            sort_by=metric_col,
            ascending=False,
        )
        mode_df["testing_mode"] = mode_label
        frames.append(mode_df)

    if not frames:
        raise FileNotFoundError(
            f"No run summaries ('{filename}') found under any mode subfolder of {base_dir}."
        )

    long_df = pd.concat(frames, ignore_index=True)

    pivot_df = long_df.pivot_table(
        index=index_col,
        columns="testing_mode",
        values=metric_col,
        aggfunc="mean",
    )

    # Keep the requested mode column order (only those that actually appeared).
    ordered_cols = [m for m in modes.keys() if m in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]
    pivot_df.columns.name = None

    if sort_by_mode is not None and sort_by_mode in pivot_df.columns:
        pivot_df = pivot_df.sort_values(sort_by_mode, ascending=False)
    else:
        pivot_df = pivot_df.sort_index()

    if decimals is not None:
        pivot_df = pivot_df.round(decimals)

    # index=False on write, so the run identifier moves into a column first.
    table_paths = save_output(
        None,
        analysis="correctness_prediction",
        plot=table_filename,
        tables={"pivot": pivot_df.reset_index()},
        save=save_table,
        to_paper=to_paper,
        subdir=subdir,
    ).paths

    return {
        "pivot_df": pivot_df,
        "long_df": long_df,
        "table_paths": table_paths,
    }