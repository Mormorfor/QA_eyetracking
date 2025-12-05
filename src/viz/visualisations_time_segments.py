import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con


# ---------------------------------------------------------------------------
# Time-Segment Analyses (before / during / after selected answer first encountered)
# ---------------------------------------------------------------------------

def _assign_time_segment(
    group: pd.DataFrame,
    area_col: str,
    selected_col: str,
    segment_col: str = Con.SEGMENT_COLUMN,
) -> pd.Series:
    """
    Helper: given all rows for one (participant, trial), assign a time segment
    ('before', 'during', 'after') based on when the selected answer is fixated.

    Logic:
      - area_col contains e.g. 'question', 'answer_A', 'answer_B', ...
      - selected_col contains 'A' / 'B' / 'C' / 'D'
      - 'before': all rows before first fixation on the selected answer
      - 'during': all fixations on the selected answer from its first
                  occurrence onwards
      - 'after' : only NON-selected areas from the first non-selected
                  fixation after the selected answer onwards
                  (later revisits to the selected answer stay 'during')
    """
    area = group[area_col].astype(str).str.lower().to_numpy()
    selected = str(group[selected_col].iloc[0]).lower()
    target = f"answer_{selected}"

    mask = (area == target)
    n = len(group)
    out = np.full(n, "before", dtype=object)

    # No fixation on the selected answer at all -> everything 'before'
    if not mask.any():
        return pd.Series(out, index=group.index, name=segment_col)

    first_pos = np.flatnonzero(mask)[0]

    out[first_pos:] = "during"

    # Look for the first NON-target after that
    after_first = mask[first_pos + 1:]
    non_target_rel = np.flatnonzero(~after_first)

    if non_target_rel.size > 0:
        interruption = first_pos + 1 + non_target_rel[0]

        # From interruption onwards, mark ONLY non-selected areas as "after".
        # Any later visits to the selected answer remain "during".
        after_range = np.arange(interruption, n)
        non_target_after = after_range[~mask[interruption:]]
        out[non_target_after] = "after"

    return pd.Series(out, index=group.index, name=segment_col)




def add_time_segment_column(
    df: pd.DataFrame,
    group_cols=(Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    selected_col: str = Con.SELECTED_ANSWER_LABEL_COLUMN,
    segment_col: str = Con.SEGMENT_COLUMN,
) -> pd.DataFrame:
    """
    Add a SEGMENT_COLUMN column to a row-level DF with IA features.

    Parameters
    ----------
    df : DataFrame
        Row-level data (one row per IA).
    group_cols : tuple[str]
        Columns that uniquely identify a trial, e.g.
        (PARTICIPANT_ID, TRIAL_ID).
    area_col : str
        Column with area labels such as 'question', 'answer_A', ...
    selected_col : str
        Column with the selected answer label ('A','B','C','D').
    segment_col : str
        Name of the resulting time-segment column.

    Returns
    -------
    df_out : DataFrame
        Copy of df with an added categorical column `segment_col`
        taking values in ['before', 'during', 'after'].
    """
    df_out = df.copy()

    df_out[segment_col] = (
        df_out
        .groupby(list(group_cols), group_keys=False)
        .apply(
            lambda g: _assign_time_segment(
                g,
                area_col=area_col,
                selected_col=selected_col,
                segment_col=segment_col,
            )
        )
    )

    order = ["before", "during", "after"]
    df_out[segment_col] = pd.Categorical(
        df_out[segment_col],
        categories=order,
        ordered=True,
    )
    return df_out


def _plot_time_segment_bar(
    df: pd.DataFrame,
    value_col: Optional[str],
    metric_col_name: str,
    metric_label: str,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    transform=None,
    subdir: str = "",
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "../reports/plots/time_segments",
    title: Optional[str] = None,
):
    """
    Generic helper to make time-segment barplots.

    Parameters
    ----------
    df : DataFrame
        Row-level data with a time-segment column (SEGMENT_COLUMN).
    value_col : str or None
        If not None, we aggregate the mean of this column per
        (participant, trial, segment). If None, we use the group size
        (.size()) as the metric.
    metric_col_name : str
        Name of the metric column in the aggregated DataFrame.
    metric_label : str
        Label to show on the y-axis and in the title.
    id_col : str
        Participant ID column.
    trial_index_col : str
        Trial ID column.
    segment_col : str
        Column indicating the time segment (e.g. SEGMENT_COLUMN).
    transform : callable or None
        Optional function applied to df before aggregation
        (e.g., to add a 'skipped' column).
    subdir : str
        Subdirectory under output_root for saving plots.
    figsize : tuple
        Figure size.
    h_or_g : str
        Group label for titles/filenames ('hunters' / 'gatherers').
    save : bool
        If True, save the figure as PNG.
    output_root : str
        Root directory for plot saving.
    title : str or None
        Custom title; if None, a default is used.

    Returns
    -------
    fig : Figure
    summary : DataFrame
        Mean, SD, and N per segment.
    """
    order = ["before", "during", "after"]

    if transform is not None:
        df = transform(df.copy())

    if value_col is None:
        df_metric = (
            df
            .groupby([id_col, trial_index_col, segment_col], observed=False)
            .size()
            .reset_index(name=metric_col_name)
        )
    else:
        df_metric = (
            df
            .groupby([id_col, trial_index_col, segment_col], observed=False)[value_col]
            .mean()
            .reset_index()
            .rename(columns={value_col: metric_col_name})
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_metric,
        x=segment_col,
        y=metric_col_name,
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("Time segment")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} by time segment ({h_or_g})")

    summary = (
        df_metric
        .groupby(segment_col)[metric_col_name]
        .agg(mean="mean", sd="std", n="count")
        .reindex(order)
        .reset_index()
    )

    fig.tight_layout()
    if save:
        out_dir = os.path.join(output_root, subdir or metric_col_name)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{h_or_g}__{subdir or metric_col_name}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=300)

    plt.show()
    return fig, summary



def plot_time_segment_mean_dwell(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "../reports/plots/time_segments",
    title: Optional[str] = None,
):
    """
    Mean dwell time (per trial) by time segment ('before', 'during', 'after').
    """
    return _plot_time_segment_bar(
        df=df,
        value_col=dwell_col,
        metric_col_name=dwell_col,
        metric_label=f"Mean {dwell_col}",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=None,
        subdir="mean_dwell_time",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )


def plot_time_segment_sequence_length(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "../reports/plots/time_segments",
    title: Optional[str] = None,
):
    """
    Average sequence length (number of IA rows) by time segment.
    """
    return _plot_time_segment_bar(
        df=df,
        value_col=None,
        metric_col_name=Con.SEQUENCE_LENGTH_COLUMN,
        metric_label="Average number of rows",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=None,
        subdir="sequence_length",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )



def plot_time_segment_fixation_count(
    df: pd.DataFrame,
    fix_col: str = Con.IA_FIXATIONS_COUNT,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "../reports/plots/time_segments",
    title: Optional[str] = None,
):
    """
    Mean fixation count (per trial) by time segment.
    """
    return _plot_time_segment_bar(
        df=df,
        value_col=fix_col,
        metric_col_name=fix_col,
        metric_label=f"Mean {fix_col}",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=None,
        subdir="fixation_count",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )


def plot_time_segment_skip_rate(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "../reports/plots/time_segments",
    title: Optional[str] = None,
):
    """
    Mean skip rate (proportion of IA rows with dwell == 0) by time segment.
    """
    def _add_skipped(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        # IA-level flag for time-segment computation
        d[Con.SKIPPED_COLUMN] = (d[dwell_col] == 0).astype(int)
        return d

    return _plot_time_segment_bar(
        df=df,
        value_col=Con.SKIPPED_COLUMN,
        metric_col_name=Con.SKIP_RATE,          # <- existing "skip_rate"
        metric_label="Skip rate (proportion)",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=_add_skipped,
        subdir="skip_rate",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )




def run_all_time_segment_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    group_cols=(Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    selected_col: str = Con.SELECTED_ANSWER_LABEL_COLUMN,
    dwell_col: str = Con.IA_DWELL_TIME,
    fix_col: str = Con.IA_FIXATIONS_COUNT,
    segment_col: str = Con.SEGMENT_COLUMN,
    output_root: str = "../reports/plots/time_segments",
    save: bool = True,
) -> dict:
    """
    Convenience wrapper:
    - add SEGMENT_COLUMN to hunters & gatherers
    - plot:
        * mean dwell time by segment
        * average sequence length by segment
        * mean fixation count by segment
        * mean skip rate by segment

    Returns
    -------
    results : dict
        results["hunters"] / results["gatherers"] each contain:
            {
              "df": df_with_segment,
              "mean_dwell": (fig, summary_df),
              "sequence_length": (fig, summary_df),
              "fixation_count": (fig, summary_df),
              "skip_rate": (fig, summary_df),
            }
    """
    results = {}

    for group_name, df in {"hunters": hunters, "gatherers": gatherers}.items():
        df_seg = add_time_segment_column(
            df,
            group_cols=group_cols,
            area_col=area_col,
            selected_col=selected_col,
            segment_col=segment_col,
        )

        md_fig, md_summary = plot_time_segment_mean_dwell(
            df_seg,
            dwell_col=dwell_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        sl_fig, sl_summary = plot_time_segment_sequence_length(
            df_seg,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        fc_fig, fc_summary = plot_time_segment_fixation_count(
            df_seg,
            fix_col=fix_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        sr_fig, sr_summary = plot_time_segment_skip_rate(
            df_seg,
            dwell_col=dwell_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        results[group_name] = {
            "df": df_seg,
            "mean_dwell": (md_fig, md_summary),
            "sequence_length": (sl_fig, sl_summary),
            "fixation_count": (fc_fig, fc_summary),
            "skip_rate": (sr_fig, sr_summary),
        }

    return results
