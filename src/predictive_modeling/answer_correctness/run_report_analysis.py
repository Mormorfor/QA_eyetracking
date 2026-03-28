from typing import Iterable, List, Optional, Sequence, Union, Dict, Any, Tuple, Mapping
from pathlib import Path


from predictive_modeling.answer_correctness.answer_correctness_viz import plot_correctness_run_comparison, \
    collect_correctness_run_reports
from viz.plot_output import save_df_csv


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
    save_table: bool = False,
    save_plot_figure: bool = False,
    rel_dir: Optional[str] = None,
    table_filename: str = "all_run_summaries",
    plot_filename: str = "run_comparison_balanced_accuracy",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
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

    table_paths = []
    if save_table:
        if rel_dir is None:
            raise ValueError("rel_dir must be provided when save_table=True.")
        table_paths = save_df_csv(
            summary_df,
            rel_dir=rel_dir,
            filename=table_filename,
            paper_dirs=paper_dirs,
        )

    fig, plot_df, plot_paths = plot_correctness_run_comparison(
        summary_df=summary_df,
        metric_col=metric_col,
        label_col=label_col,
        top_n=top_n,
        figsize=figsize,
        title=title,
        save=save_plot_figure,
        rel_dir=rel_dir,
        filename=plot_filename,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return {
        "summary_df": summary_df,
        "plot_df": plot_df,
        "fig": fig,
        "table_paths": table_paths,
        "plot_paths": plot_paths,
    }