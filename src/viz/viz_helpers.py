# src/viz/viz_helpers.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as C


def split_participant_groups(
    all_participants: pd.DataFrame,
    split: bool = True,
    include_all: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Split a single ``all_participants`` DataFrame into the groups the
    visualisation modules plot over.

    The hunters/gatherers distinction is the question-preview split (mirrors
    ``data_prep.data_csv_generation.split_hunters_and_gatherers``):
      - hunters   : ``question_preview == True``
      - gatherers : ``question_preview == False``

    ``all_participants`` is expected to be the processed all-participants table
    (repeated/practice trials already removed by the generation pipeline), so
    the two halves reconcatenate to it exactly.

    Returns an insertion-ordered dict::

        {"hunters": ..., "gatherers": ..., "all_participants": ...}

    Parameters
    ----------
    split : bool, default True
        If ``False``, skip the hunters/gatherers split entirely and return a
        single group ``{"all_participants": all_participants}`` (``include_all``
        is ignored in that case).
    include_all : bool, default True
        When splitting, whether to also include the ``"all_participants"`` entry
        (the concatenation of the two halves). Ignored when ``split=False``.
    """
    if not split:
        return {"all_participants": all_participants}

    preview = all_participants[C.QUESTION_PREVIEW_COLUMN]
    hunters = all_participants[preview == True].copy()
    gatherers = all_participants[preview == False].copy()

    groups: Dict[str, pd.DataFrame] = {"hunters": hunters, "gatherers": gatherers}
    if include_all:
        groups["all_participants"] = pd.concat(
            [hunters, gatherers], ignore_index=True
        )

    # A group with no rows is not a group. It arises when this is handed a frame
    # that only holds one side of the split (hunters.csv, say) -- legitimate
    # usage, but every downstream summary would then be computed over zero
    # trials and report a result anyway: Fisher on an all-zero table returns
    # p = 1.0, not an error. Drop such groups and say so.
    empty = [name for name, frame in groups.items() if frame.empty]
    for name in empty:
        print(f"[split_participant_groups] no rows for {name!r} -- group skipped")
        del groups[name]

    return groups


def p_to_stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_wilson_errorbars_and_ns(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    y: str = "accuracy",
    n_col: str = "n",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    show_n: bool = True,
) -> None:
    """
    Draw Wilson CI error bars and (optionally) add n=... labels above bars.

    Expects columns:
      - y (default: 'accuracy')
      - ci_low_col (default: 'ci_low')
      - ci_high_col (default: 'ci_high')
      - n_col (default: 'n') if show_n=True
    """
    for i, r in summary_df.reset_index(drop=True).iterrows():
        acc = r[y]

        if np.isfinite(r[ci_low_col]) and np.isfinite(r[ci_high_col]) and np.isfinite(acc):
            ax.errorbar(
                i,
                acc,
                yerr=[[acc - r[ci_low_col]], [r[ci_high_col] - acc]],
                fmt="none",
                capsize=4,
                ecolor="black",
                elinewidth=1.5,
            )

        if show_n and np.isfinite(acc):
            ax.text(
                i,
                min(0.98, acc + 0.03),
                f"n={int(r[n_col])}",
                ha="center",
                va="bottom",
            )


def add_significance_bracket(ax: plt.Axes, stars: str, x1: int = 0, x2: int = 1) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    # pick a y above max bar height
    heights = [p.get_height() for p in ax.patches]  # bars
    y_max = float(np.nanmax(heights)) if heights else 0.0

    y = y_max + 0.06 * yr
    h = 0.03 * yr

    # expand ylim so bracket fits
    if y + h + 0.05 * yr > ymax:
        ax.set_ylim(ymin, y + h + 0.08 * yr)

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.01 * yr, stars, ha="center", va="bottom", color="black", clip_on=False)


def barplot_accuracy(summary_df: pd.DataFrame, order: Sequence[str], figsize=(6, 4)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=summary_df, x="group", y="accuracy", order=list(order), ax=ax)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, ax


def correctness_tables(
    summary_df: pd.DataFrame, test_res: Optional[Dict]
) -> Dict[str, object]:
    """Assemble the ``tables=`` payload shared by the correctness-association plots.

    Replaces the old ``save_plot_and_report``, which was a second save path
    alongside ``plot_output`` (``docs/todo.md`` T1.5). Saving now belongs to
    ``plot_output.save_output``; this only decides *what* travels with the figure.
    """
    tables: Dict[str, object] = {"summary": summary_df}
    if test_res is not None:
        tables["fisher"] = dict(test_res)
    return tables
