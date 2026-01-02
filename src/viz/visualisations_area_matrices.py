import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con



# ---------------------------------------------------------------------------
#  Base Statistics Heatmaps
# ---------------------------------------------------------------------------

def matrix_plot_ABCD(
    df: pd.DataFrame,
    stat: str,
    selected: str = "A",
    h_or_g: str = "hunters",
    drop_questions: bool = True,
    output_root: str = "../reports/plots/basic_stats_heatmaps",
    show: bool = True,
    save: bool = True,
) -> None:
    """
    Draw a heatmap of a metric by (area_label x area_screen_loc)
    for participants who selected a given answer label (A/B/C/D).

    Parameters
    ----------
    df : DataFrame
        Row-level data already filtered to the desired subset
        (e.g. only trials where selected_answer_label == 'A').
    stat : str
        Column name of the metric to visualize (e.g. 'mean_dwell_time')
        Should be selected from C.AREA_METRIC_COLUMNS
    selected : str, optional
        Which answer label was selected ('A', 'B', 'C', 'D').
    h_or_g : str, optional
        Tag for hunters/gatherers, used in the plot title and filename.
    drop_questions : bool, optional
        If True, exclude rows where AREA_LABEL_COLUMN == 'question'.
    output_root : str, optional
        Root directory where plots will be saved.
    show : bool, optional
        If True, display the plot.
    save : bool, optional
        If True, save the plot as a PNG file under output_root/stat/.
    """
    df = df[
        [Con.TRIAL_ID, Con.PARTICIPANT_ID, Con.AREA_LABEL_COLUMN, Con.AREA_SCREEN_LOCATION, stat]
    ].drop_duplicates().copy()

    if drop_questions:
        df = df[df[Con.AREA_LABEL_COLUMN] != "question"]

    matrix = pd.pivot_table(
        data=df,
        index=Con.AREA_LABEL_COLUMN,
        columns=Con.AREA_SCREEN_LOCATION,
        values=stat,
        aggfunc="mean",
    )

    if drop_questions:
        label_order = [lbl for lbl in Con.ANSWER_LABEL_CHOICES if lbl != "question"]
    else:
        label_order = list(Con.ANSWER_LABEL_CHOICES)

    row_order = [lbl for lbl in label_order if lbl in matrix.index]
    col_order = [loc for loc in Con.AREA_LABEL_CHOICES if loc in matrix.columns]

    matrix = matrix.reindex(index=row_order, columns=col_order)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar_kws={"label": stat},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    title_suffix = " (questions removed)" if drop_questions else ""
    plt.title(f"{stat} of those who chose {selected}{title_suffix}")
    plt.xlabel(Con.AREA_SCREEN_LOCATION)
    plt.ylabel(Con.AREA_SCREEN_LOCATION)
    plt.tight_layout()

    if save:
        out_dir = os.path.join(output_root, h_or_g, stat)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{selected}_selected.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()



def label_vs_loc_mat(
    metric: str,
    dfh: pd.DataFrame,
    dfg: pd.DataFrame,
    dfa: Optional[pd.DataFrame] = None,
    drop_questions: bool = False,
    **plot_kwargs,
) -> None:
    """
    For a given metric, plot heatmaps for each selected answer label (A–D).

    Runs:
      - hunters
      - gatherers
      - all participants (optional, if dfa is provided)
    """
    def _run(df: pd.DataFrame, group_label: str) -> None:
        print(f"{group_label.upper()} (drop_questions={drop_questions})")
        for ans in ["A", "B", "C", "D"]:
            subset = df[df[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans]
            matrix_plot_ABCD(
                subset,
                metric,
                selected=ans,
                h_or_g=group_label,          # used in filename
                drop_questions=drop_questions,
                **plot_kwargs,
            )

    _run(dfh, "hunters")
    _run(dfg, "gatherers")

    if dfa is not None:
        _run(dfa, "all participants")




def run_all_area_metric_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    metrics=None,
    drop_question_variants=(False, True),
    output_root="../reports/plots/basic_stats_heatmaps",
    show=True,
    save=True,
):
    """
    Convenience wrapper: for every metric, generate label-vs-location matrices
    for hunters, gatherers, and all participants, with and without questions.

    """
    if metrics is None:
        metrics = Con.AREA_METRIC_COLUMNS

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    for metric in metrics:
        for dq in drop_question_variants:
            dq_folder = "questions_removed" if dq else "questions_included"

            print(f"\n=== {metric} (drop_questions={dq}) ===")

            label_vs_loc_mat(
                metric,
                hunters,
                gatherers,
                dfa=all_participants,
                drop_questions=dq,
                output_root=os.path.join(output_root, dq_folder),
                show=show,
                save=save,
            )

