"""Splitting Study 1's participants into hunters and gatherers.

Both functions split on `question_preview` (True = hunters, False =
gatherers), but they are not duplicates:

  split_hunters_and_gatherers  for raw data: also drops repeated-reading and
                               practice trials, and returns (hunters, gatherers)
  split_participant_groups     for processed data, where those trials are
                               already gone: returns {name: frame}, optionally
                               with "all_participants" too
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.config import columns as C


def split_hunters_and_gatherers(df, remove_repeats=True, remove_practice=True):
    """Split raw trials into hunters and gatherers, after dropping
    repeated-reading and practice trials (both on by default)."""
    df_filtered = df.copy()
    if remove_repeats:
        df_filtered = df_filtered[df_filtered[C.REPEATED_TRIAL_COLUMN] == False].copy()
    if remove_practice:
        df_filtered = df_filtered[df_filtered[C.PRACTICE_TRIAL_COLUMN] == False].copy()

    df_hunters = df_filtered[df_filtered[C.QUESTION_PREVIEW_COLUMN] == True].copy()
    df_gatherers = df_filtered[df_filtered[C.QUESTION_PREVIEW_COLUMN] == False].copy()

    return df_hunters, df_gatherers


def split_participant_groups(
    all_participants: pd.DataFrame,
    split: bool = True,
    include_all: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Split processed data into the groups the figures are drawn for.

    Returns ``{"hunters": ..., "gatherers": ..., "all_participants": ...}``,
    where all_participants is the two halves put back together.

    - ``split=False`` returns only ``{"all_participants": ...}``.
    - ``include_all=False`` leaves out "all_participants".

    A group with no rows is dropped, with a printed note.
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

    # An empty group happens when the frame holds only one side of the split
    # (only hunters, say). Kept, it would still produce results -- a Fisher
    # test on an all-zero table returns p = 1.0, not an error. So it is dropped,
    # with a note.
    empty = [name for name, frame in groups.items() if frame.empty]
    for name in empty:
        print(f"[split_participant_groups] no rows for {name!r} -- group skipped")
        del groups[name]

    return groups
