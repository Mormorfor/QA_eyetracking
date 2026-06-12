"""
visualisations_last_label.py

Frequency of the last area label fixated before confirming an answer
(`last_lbl_before_confirm`), split into one subplot per selected answer (A-D).

Input: area-level dataframe (multiple rows per trial). The last-label feature is
trial-level, so rows are de-duplicated to one per trial before counting.

Plots:
  - 2x2 grid of bar charts (one per selected answer) showing either the raw
    trial counts or the in-group proportion of each `last_lbl_before_confirm`
    label (toggled via `normalize`).

All on-plot text is presentation-clean (no underscores / variable names). The
hunters / gatherers / all-participants distinction appears only in saved
filenames, never on the figure.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con
from src.viz.plot_output import save_plot


_LABEL_ORDER = ["question", "answer_A", "answer_B", "answer_C", "answer_D"]
_SELECTED_ANSWERS = ["A", "B", "C", "D"]

# Presentation-clean display names for the area labels (x-axis ticks).
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
    """Collapse the area-level df to one row per trial, keeping the trial-level
    selected-answer and last-label columns."""
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
    save: bool = False,
    paper_dirs=None,
    h_or_g: str = "all_participants",
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot the frequency of `last_lbl_before_confirm` as four subplots, one per
    selected answer (A/B/C/D).

    Parameters
    ----------
    normalize : bool
        True  -> in-group proportion (each selected-answer panel sums to 1).
        False -> raw counts of trials.
    title, x_label, y_label : Optional[str]
        On-plot text. `y_label=None` auto-selects a clean default based on
        `normalize` ("Proportion of Trials" / "Number of Trials"). Pass an empty
        string to suppress a label entirely, or None for the default.
    panel_title_fmt : str
        Format string for each subplot title; `{ans}` is the answer letter.
    show_n : bool
        If True, append the per-panel sample size as "(n = 1,234)".
    h_or_g : str
        Group tag used ONLY in the saved filename, never shown on the figure.

    If save=True, always saves to:
        reports/plots/last_label_before_confirm/<h_or_g>__<prop|count>.png
    If paper_dirs is a list, also mirrors there (see save_plot).

    Returns
    -------
    (fig, summary_df) where summary_df has columns:
        selected_answer, last_lbl_before_confirm, n, proportion
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

    if save:
        save_plot(
            fig=fig,
            rel_dir="last_label_before_confirm",
            filename=f"{h_or_g}__{'prop' if normalize else 'count'}",
            ext="png",
            dpi=300,
            paper_dirs=paper_dirs,
        )

    return fig, summary_df


def run_all_last_label_before_confirm_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    normalize: bool = True,
    title: Optional[str] = "Last Area Viewed Before Confirming an Answer",
    x_label: Optional[str] = "Last Area Viewed",
    y_label: Optional[str] = None,
    panel_title_fmt: str = "Chose {ans}",
    show_n: bool = True,
    save_plots: bool = True,
    paper_dirs=None,
    print_summaries: bool = False,
) -> Dict:
    """
    Produce the last-label-before-confirm frequency plot for hunters, gatherers,
    and all participants combined.

    Titles / axis labels are shared across groups (the group never appears on the
    figure, only in the saved filename). See `plot_last_label_before_confirm_freq`
    for the meaning of the text parameters.

    Returns
    -------
    results[group] = {"fig": Figure, "summary": DataFrame}
    """
    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    groups = {
        "hunters": hunters,
        "gatherers": gatherers,
        "all_participants": all_participants,
    }

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
            save=save_plots,
            paper_dirs=paper_dirs,
            h_or_g=group_key,
        )
        if print_summaries:
            print(f"\n=== {group_key.upper()} ===")
            print(summary)
        results[group_key] = {"fig": fig, "summary": summary}

    return results
