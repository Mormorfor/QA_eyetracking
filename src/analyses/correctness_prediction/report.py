"""Comparing saved runs of the headline model.

Each run of a `run.py` bundle with `save=True` writes its scores as a one-row table,
`model_summary__<run and model>__summary.csv`. The functions here find those files
and compare the runs: as a bar chart of one score (`collect_and_plot_correctness_runs`),
or as a table with one column per held-out regime (`collect_correctness_runs_by_mode`).
Each run is one train/test split; comparing cross-validated models is
`plots/run_comparison.py::plot_cv_model_comparison_staged`.

The functions scan the folders they are given rather than taking a list of runs,
so every run ever saved there is included. Check the folder before making a final
figure.
"""

from typing import Iterable, List, Optional, Sequence, Union, Dict, Any, Tuple, Mapping
from pathlib import Path

import pandas as pd

from src.analyses.correctness_prediction.plots.run_comparison import (
    collect_correctness_run_reports,
    plot_correctness_run_comparison,
)
from src.lib.plotting.output import save_output


def collect_and_plot_correctness_runs(
    report_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    filename: str = "model_summary*__summary.csv",
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
    Collect every saved run summary under `report_dirs` and draw them as one bar
    chart of `metric_col`, highest at the top.

    The collecting is `plots/run_comparison.py::collect_correctness_run_reports`
    (`filename`, `recursive`, `sort_by`, `ascending`) and the chart is
    `plot_correctness_run_comparison` (everything else). Returns the collected
    table, the plotted rows, the figure and the saved paths.
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
        # One save writes the figure and its table, so there is one list of paths.
        "saved_paths": plot_paths,
    }


def collect_correctness_runs_by_mode(
    base_dir: Union[str, Path],
    modes: Optional[Mapping[str, str]] = None,
    filename: str = "model_summary*__summary.csv",
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
    A table of one score (balanced accuracy by default) for every saved run, with
    one column per held-out regime ("testing mode").

    `base_dir` holds one folder per mode, named as `run.py` names its output
    folders: e.g. `reports/correctness_prediction/tables/`, with `new_item/`,
    `new_subject/` and `both/` inside. `modes` maps each column label to its folder
    (by default each of the three regimes to its own name), and the columns keep
    its order. A missing folder is skipped with a printed note; a folder with no
    run summaries in it stops with an error.

    One row per run, labelled by `index_col` (`run_identifier`). Runs that share a
    label within a mode are averaged into one cell, and a run saved without a label
    is left out. Rows are sorted by the `sort_by_mode` column, highest first, if
    that column exists, otherwise by label; `decimals` rounds the cells.

    With `save_table=True` the table is saved as `<table_filename>__pivot.csv`, in
    `reports/correctness_prediction/tables/<subdir>/`.

    Returns a dict: `pivot_df` (the table), `long_df` (every run summary, with a
    `testing_mode` column) and `table_paths` (empty unless saved).
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

    # The columns in the order of `modes`, for the modes that were found.
    ordered_cols = [m for m in modes.keys() if m in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]
    pivot_df.columns.name = None

    if sort_by_mode is not None and sort_by_mode in pivot_df.columns:
        pivot_df = pivot_df.sort_values(sort_by_mode, ascending=False)
    else:
        pivot_df = pivot_df.sort_index()

    if decimals is not None:
        pivot_df = pivot_df.round(decimals)

    # Tables are written without their index, so the run label becomes a column first.
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