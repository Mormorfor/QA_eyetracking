"""Which participants a quantity is computed over.

The two functions here split the same way -- on `question_preview` -- and are
**not** duplicates, which is the thing to notice before anyone tries to merge
them:

  split_hunters_and_gatherers  pipeline-time. Returns a (hunters, gatherers)
                               tuple and does the repeated/practice filtering
                               itself, because it runs before the processed
                               table exists.
  split_participant_groups     plot-time. Returns a dict of named groups,
                               optionally including "all_participants", and
                               assumes filtering already happened.

Both were moved here in stage C (map section 5.2) because grouping is a domain
concept, not a plotting one -- `split_participant_groups` lived in viz_helpers,
where its `include_all=False` default is what hides the all-participants
strategy figure (`findings.md` section 1.2).

This is also where the T3.21 scope machinery belongs if it ever needs a home
outside features/strategies.py.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.config import columns as C


def split_hunters_and_gatherers(df, remove_repeats=True, remove_practice=True):
    """
    Split trials into 'hunters' and 'gatherers' based on question preview.
    Optionally removes repeated and practice trials before splitting.

    """
    df_filtered = df.copy()
    if remove_repeats:
        df_filtered = df_filtered[df_filtered[C.REPEATED_TRIAL_COLUMN] == False].copy()
    if remove_practice:
        df_filtered = df_filtered[df_filtered[C.PRACTICE_TRIAL_COLUMN] == False].copy()

    df_hunters = df_filtered[df_filtered[C.QUESTION_PREVIEW_COLUMN] == True].copy()
    df_gatherers = df_filtered[df_filtered[C.QUESTION_PREVIEW_COLUMN] == False].copy()

    return df_hunters, df_gatherers


# ---------------------------------------------------------------------------
#  Basic per Row Features Creation
# ---------------------------------------------------------------------------


def split_participant_groups(
    all_participants: pd.DataFrame,
    split: bool = True,
    include_all: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Split a single ``all_participants`` DataFrame into the groups the
    visualisation modules plot over.

    The hunters/gatherers distinction is the question-preview split (mirrors
    ``features.scope.split_hunters_and_gatherers``):
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
