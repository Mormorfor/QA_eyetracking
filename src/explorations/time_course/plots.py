"""Does behaviour change once the chosen answer has been encountered? Unfinished.

The idea: split each trial at the participant's first encounter with the answer
they went on to choose, and compare the parts before, during and after it. Four
measurements are compared across the parts -- mean dwell time per word, mean
fixation count per word, skip rate (the share of words never looked at) and the
number of word rows -- as one bar chart per measurement and group, each saved
with a summary table under `reports/explorations/time_course/`. Run from
`notebooks/exploration/time_course.ipynb`.

Not part of the paper, and not finished: **the split does not yet follow time.**
It goes by the order of the rows, and the word table lists a trial's words by
`IA_ID`: the question first, then the answers at the top, left, right and bottom
(on Study 1, in all but 10 trials). So "before" is the question and the answers
that come before the chosen one in that order, "during" the chosen answer's own
words, and "after" the answers that come after it (none, when the chosen answer
is at the bottom). On Study 1, in 24% of trials a word counted as "after" was first looked at before
the chosen answer was. Splitting by time would need each trial's fixations in
time order, from the fixation report.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.features.scope import split_participant_groups


# ---------------------------------------------------------------------------
# Segments: before / during / after the chosen answer
# ---------------------------------------------------------------------------

def _assign_time_segment(
    group: pd.DataFrame,
    area_col: str,
    selected_col: str,
    segment_col: str = Con.SEGMENT_COLUMN,
) -> pd.Series:
    """
    Label one trial's rows "before", "during" or "after", by their order (the
    rule is in `add_time_segment_column`).
    """
    area = group[area_col].astype(str).str.lower().to_numpy()
    selected = str(group[selected_col].iloc[0]).lower()
    target = f"answer_{selected}"

    mask = (area == target)
    n = len(group)
    out = np.full(n, "before", dtype=object)

    # The chosen answer has no rows: the whole trial is "before".
    if not mask.any():
        return pd.Series(out, index=group.index, name=segment_col)

    first_pos = np.flatnonzero(mask)[0]

    out[first_pos:] = "during"

    # The first row after that which is not on the chosen answer
    after_first = mask[first_pos + 1:]
    non_target_rel = np.flatnonzero(~after_first)

    if non_target_rel.size > 0:
        interruption = first_pos + 1 + non_target_rel[0]

        # From there on, rows not on the chosen answer are "after"; rows on the
        # chosen answer stay "during".
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
    Add a `time_segment` column ("before", "during", "after") to a table with
    several rows per trial, such as the word table.

    Within each trial (`group_cols`), the rows are taken in the order they come
    in: rows before the chosen answer's first row are "before", every row of the
    chosen answer is "during", and every other row after that first one is
    "after". The chosen answer is `answer_<letter>` in `area_col`, with the
    letter from `selected_col`; a trial whose chosen answer has no rows is all
    "before". The split follows row order, not time -- see the module
    docstring.

    Returns a copy with the column added, as an ordered category. Stops with
    an error on an empty frame. A frame holding a single trial fails too, with
    a KeyError (see the comment in the code).
    """
    df_out = df.copy()

    if df_out.empty:
        raise ValueError(
            "add_time_segment_column got zero rows. Segments are defined relative "
            "to the first fixation on the selected answer, which does not exist "
            "for an empty frame."
        )

    # With several trials, pandas returns one Series covering every row. With a
    # single trial it returns a one-row DataFrame whose columns are the row
    # labels, so the column lookup below fails with a KeyError.
    segments = (
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
    if isinstance(segments, pd.DataFrame):
        segments = segments[segment_col]
    df_out[segment_col] = segments

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
    save: Optional[bool] = None,
    to_paper=None,
    title: Optional[str] = None,
):
    """
    Bar chart of one measurement across the three segments, for one group.

    For each trial and segment, takes the mean of `value_col` over the
    segment's rows, or the number of rows when `value_col` is None. Each bar is
    the mean over trials, with a 95% bootstrap interval that treats trials as
    independent. A trial with no rows in a segment counts as 0 in the row count,
    and is left out of that segment's bar for the other measurements.

    `transform` is applied to a copy of `df` first (e.g. to add a column). Saves
    the figure and a summary table (mean, SD and number of trials per segment)
    under `reports/explorations/time_course/`, in `subdir` (default: the metric
    name).
    Returns `(fig, summary)`.
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

    save_output(
        fig,
        analysis="explorations/time_course",
        plot="time_segment_bar",
        tables={"summary": summary},
        save=save,
        to_paper=to_paper,
        subdir=subdir or metric_col_name,
        group=h_or_g,
        metric=subdir or metric_col_name,
    )

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
    save: Optional[bool] = None,
    to_paper=None,
    title: Optional[str] = None,
):
    """
    Mean dwell time per word in each segment, averaged over trials. Words never
    looked at count as 0.
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
        to_paper=to_paper,
        title=title,
    )


def plot_time_segment_ia_row_count(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: Optional[bool] = None,
    to_paper=None,
    title: Optional[str] = None,
):
    """
    Number of word rows in each segment, averaged over trials.

    This counts the words in the segment, looked at or not, so on the word table
    it depends only on the text on screen and on which answer was chosen. It is
    not `seq_len`, the number of visits in the trial table.
    """
    return _plot_time_segment_bar(
        df=df,
        value_col=None,
        metric_col_name=Con.SEGMENT_IA_ROW_COUNT,
        metric_label="Average number of rows",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=None,
        subdir="segment_ia_row_count",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        to_paper=to_paper,
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
    save: Optional[bool] = None,
    to_paper=None,
    title: Optional[str] = None,
):
    """
    Mean fixation count per word in each segment, averaged over trials.
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
        to_paper=to_paper,
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
    save: Optional[bool] = None,
    to_paper=None,
    title: Optional[str] = None,
):
    """
    Skip rate in each segment: the share of the segment's words with a dwell
    time of 0, averaged over trials.
    """
    def _add_skipped(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        # 1 if the word was never looked at
        d[Con.SKIPPED_COLUMN] = (d[dwell_col] == 0).astype(int)
        return d

    return _plot_time_segment_bar(
        df=df,
        value_col=Con.SKIPPED_COLUMN,
        metric_col_name=Con.SKIP_RATE,
        metric_label="Skip rate (proportion)",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=_add_skipped,
        subdir="skip_rate",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        to_paper=to_paper,
        title=title,
    )




def run_all_time_segment_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    group_cols=(Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    selected_col: str = Con.SELECTED_ANSWER_LABEL_COLUMN,
    dwell_col: str = Con.IA_DWELL_TIME,
    fix_col: str = Con.IA_FIXATIONS_COUNT,
    segment_col: str = Con.SEGMENT_COLUMN,
    save: Optional[bool] = None,
    to_paper=None,
) -> dict:
    """
    Add the segments and draw the four charts (mean dwell time, word-row count,
    mean fixation count, skip rate), for hunters, gatherers and all participants
    together (only the last with `split_groups=False`).

    Returns `{group: {...}}`, each with `df` (the frame with the segment column)
    and a `(fig, summary)` pair under "mean_dwell", "segment_ia_row_count",
    "fixation_count" and "skip_rate".
    """
    results = {}

    groups = split_participant_groups(all_participants, split=split_groups)

    for group_key, df in groups.items():
        # Pretty label for titles/filenames
        group_label = "all participants" if group_key == "all_participants" else group_key

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
            h_or_g=group_label,
            save=save,
            to_paper=to_paper,
        )

        sl_fig, sl_summary = plot_time_segment_ia_row_count(
            df_seg,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_label,
            save=save,
            to_paper=to_paper,
        )

        fc_fig, fc_summary = plot_time_segment_fixation_count(
            df_seg,
            fix_col=fix_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_label,
            save=save,
            to_paper=to_paper,
        )

        sr_fig, sr_summary = plot_time_segment_skip_rate(
            df_seg,
            dwell_col=dwell_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_label,
            save=save,
            to_paper=to_paper,
        )

        results[group_key] = {
            "df": df_seg,
            "mean_dwell": (md_fig, md_summary),
            "segment_ia_row_count": (sl_fig, sl_summary),
            "fixation_count": (fc_fig, fc_summary),
            "skip_rate": (sr_fig, sr_summary),
        }

    return results

