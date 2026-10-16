"""
The last area looked at before confirming an answer, split by the answer chosen.

Backs the paper's End of trial behavior subsection. A trial's
`last_lbl_before_confirm` is the area (the question or one of the four answers)
of the last fixation that started at or before the confirming press
(`glossary.md` §9). The figure is a 2x2 grid with one panel per chosen answer
(A-D) and one bar per area: how many of those trials, or what share of them,
ended on that area.

The input may have several rows per trial (e.g. the word table); it is reduced
to one row per trial before counting.

The figure text uses plain names and does not show the group: hunters,
gatherers and all participants are told apart by the file name only.
"""

from __future__ import annotations

from typing import Optional, Dict, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.features.scope import split_participant_groups


_LABEL_ORDER = ["question", "answer_A", "answer_B", "answer_C", "answer_D"]
_SELECTED_ANSWERS = ["A", "B", "C", "D"]

# Display names for the area labels on the x-axis.
_LABEL_DISPLAY = {
    "question": "Question",
    "answer_A": "Answer A",
    "answer_B": "Answer B",
    "answer_C": "Answer C",
    "answer_D": "Answer D",
}


def _dedup_to_trials(
    df: pd.DataFrame,
    label_col: str,
    selected_col: str,
    trial_cols: Sequence[str],
) -> pd.DataFrame:
    """One row per trial, with the chosen answer and the last-look column.
    Trials with no last look are dropped."""
    cols = list(trial_cols) + [selected_col, label_col]
    return (
        df[cols]
        .drop_duplicates(subset=list(trial_cols))
        .dropna(subset=[label_col])
    )


def plot_last_label_before_confirm_freq(
    df: pd.DataFrame,
    label_col: str = Con.LAST_LBL_BEFORE_CONFIRM,
    selected_col: str = Con.SELECTED_ANSWER_LABEL_COLUMN,
    trial_cols: Sequence[str] = (
        Con.PARTICIPANT_ID,
        Con.TRIAL_ID,
        Con.TEXT_ID_COLUMN,
    ),
    normalize: bool = True,
    title: Optional[str] = "Last Area Viewed Before Confirming an Answer",
    x_label: Optional[str] = "Last Area Viewed",
    y_label: Optional[str] = None,
    panel_title_fmt: str = "Chose {ans}",
    show_n: bool = True,
    figsize: Tuple[int, int] = (12, 9),
    save: Optional[bool] = None,
    to_paper=None,
    h_or_g: str = "all_participants",
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Draw the 2x2 grid for one group: in each panel (one chosen answer), how
    often each area was the last one looked at before confirming.

    With `normalize` (the default) the bars are shares of the panel's trials;
    without it, numbers of trials. `y_label=None` picks the matching label and
    "" leaves it blank. `panel_title_fmt` fills `{ans}` with the letter, and
    `show_n` adds the panel's number of trials. `h_or_g` goes into the file
    name only.

    Trials with no last look are left out. A panel whose answer nobody chose
    shows zeros.

    Saves the figure and its table under `reports/last_visitation/`. Returns
    `(fig, summary_df)`, where `summary_df` has the columns selected_answer,
    last_lbl_before_confirm, n and proportion.
    """
    trials = _dedup_to_trials(df, label_col, selected_col, trial_cols)

    label_order = [l for l in _LABEL_ORDER if l in trials[label_col].unique()]
    tick_labels = [_LABEL_DISPLAY.get(l, l) for l in label_order]

    y_col = "proportion" if normalize else "n"
    if y_label is None:
        y_label = "Proportion of Trials" if normalize else "Number of Trials"

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    summary_rows = []

    for ans, ax in zip(_SELECTED_ANSWERS, axes.ravel()):
        subset = trials[trials[selected_col] == ans]
        n_total = len(subset)

        counts = (
            subset[label_col]
            .value_counts()
            .reindex(label_order, fill_value=0)
        )
        prop = counts / n_total if n_total else counts * 0.0

        plot_df = pd.DataFrame(
            {
                label_col: label_order,
                "n": counts.values,
                "proportion": prop.values,
            }
        )

        sns.barplot(
            data=plot_df,
            x=label_col,
            y=y_col,
            order=label_order,
            ax=ax,
        )

        panel_title = panel_title_fmt.format(ans=ans)
        if show_n:
            panel_title = f"{panel_title}  (n = {n_total:,})"
        ax.set_title(panel_title, fontsize=14)

        ax.set_xlabel(x_label or "", fontsize=12)
        ax.set_ylabel(y_label or "", fontsize=12)
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=11)
        sns.despine(ax=ax)

        plot_df.insert(0, "selected_answer", ans)
        summary_rows.append(plot_df)

    if title:
        fig.suptitle(title, y=1.02, fontsize=18)
    fig.tight_layout()

    summary_df = pd.concat(summary_rows, ignore_index=True)

    save_output(
        fig,
        analysis="last_visitation",
        plot="last_label_before_confirm",
        tables={"summary": summary_df},
        save=save,
        to_paper=to_paper,
        group=h_or_g,
        scale="prop" if normalize else "count",
    )

    return fig, summary_df


def run_all_last_label_before_confirm_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    normalize: bool = True,
    title: Optional[str] = "Last Area Viewed Before Confirming an Answer",
    x_label: Optional[str] = "Last Area Viewed",
    y_label: Optional[str] = None,
    panel_title_fmt: str = "Chose {ans}",
    show_n: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
) -> Dict:
    """
    Draw the last-look grid for hunters, gatherers and all participants
    together (only the last with `split_groups=False`).

    The figure text is the same for every group; the text options are those of
    `plot_last_label_before_confirm_freq`. `print_summaries` prints each
    group's table.

    Returns `{group: {"fig": ..., "summary": ...}}`.
    """
    groups = split_participant_groups(all_participants, split=split_groups)

    results: Dict = {}
    for group_key, df in groups.items():
        fig, summary = plot_last_label_before_confirm_freq(
            df,
            normalize=normalize,
            title=title,
            x_label=x_label,
            y_label=y_label,
            panel_title_fmt=panel_title_fmt,
            show_n=show_n,
            save=save,
            to_paper=to_paper,
            h_or_g=group_key,
        )
        if print_summaries:
            print(f"\n=== {group_key.upper()} ===")
            print(summary)
        results[group_key] = {"fig": fig, "summary": summary}

    return results
