"""Which screen area each word is in, worked out from where it sits on screen.

Each word's box has a position (IA_TOP, IA_LEFT). The answer screen always has
the same layout, so position alone says whether a word belongs to the question
or to which answer.

The pipeline also assigns areas a second way, by counting words in the stored
question and answer texts. The two are compared on every run:

- where they disagree, position wins, because it is what was measured -- the
  stored text does not always match what was displayed;
- every correction is printed;
- if they disagree on more than 5% of trials, the run stops, because that means
  the layout below does not describe this dataset's screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import columns as C


# ---------------------------------------------------------------------------
#  Answer-screen layout
# ---------------------------------------------------------------------------
# The answer screen always has the same layout: the question on one or two lines
# at the top, then the four answers in a diamond -- one above, two side by side,
# one below. So every word's box falls into one of four horizontal bands, and
# the middle band is split into left and right.
#
# The boundaries are the midpoints of the gaps between bands, measured on the
# Study 1 data: bands at IA_TOP 153-265 / 381-720 / 723-1062 / 1065-1294, with
# the left answer ending at IA_LEFT 873 and the right one starting at 1717.
# Across all 759,990 Study 1 words, no word falls in two bands. Study 2 uses the
# same screen, so the same boundaries apply.
#
# The boundaries are written down rather than found from the data, because they
# cannot be found from a single trial: lines inside one area are ~110 px apart,
# while two neighbouring bands can be just 3 px apart. Gap-finding would find
# line breaks, not area boundaries.
AREA_BAND_QUESTION_TOP = 323.0


AREA_BAND_TOP_MIDDLE = 721.5


AREA_BAND_MIDDLE_BOTTOM = 1063.5


AREA_SPLIT_LEFT_RIGHT = 1295.0


# If the layout above did not fit a dataset's screen, the two ways of assigning
# areas would disagree on nearly every trial, not on a handful. Above this share
# of trials the run stops instead of relabelling everything.
MAX_GEOMETRY_DISAGREEMENT = 0.05


def assign_area_by_geometry(df: pd.DataFrame) -> np.ndarray:
    """The screen area of each word, from where its box sits on screen.

    Returns one location per row: "question", "answer_0(top)",
    "answer_1(left)", "answer_2(right)" or "answer_3(bottom)".
    """
    missing = [c for c in (C.IA_TOP, C.IA_LEFT) if c not in df.columns]
    if missing:
        raise KeyError(
            f"interest-area geometry {missing} is required to place words on the "
            "answer screen; it is present in every raw report this project reads"
        )

    top = pd.to_numeric(df[C.IA_TOP]).to_numpy()
    left = pd.to_numeric(df[C.IA_LEFT]).to_numpy()
    question, top_ans, left_ans, right_ans, bottom_ans = C.LOC_CHOICES
    return np.select(
        [
            top < AREA_BAND_QUESTION_TOP,
            top < AREA_BAND_TOP_MIDDLE,
            (top < AREA_BAND_MIDDLE_BOTTOM) & (left < AREA_SPLIT_LEFT_RIGHT),
            top < AREA_BAND_MIDDLE_BOTTOM,
        ],
        [question, top_ans, left_ans, right_ans],
        default=bottom_ans,
    )


def _reconcile_area_with_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the word-count areas with the on-screen positions, and keep the
    positions.

    `df` already has `area_screen_loc` from counting words in the stored text.
    This replaces it with the position-based area, prints how many words and
    trials changed, and stops with an error if more than 5% of trials changed.
    """
    by_counts = df[C.AREA_SCREEN_LOCATION].to_numpy()
    by_geometry = assign_area_by_geometry(df)
    disagree = by_counts != by_geometry

    n_trials = len(df.index.unique())
    corrected = df.index[disagree].unique()
    share = len(corrected) / n_trials if n_trials else 0.0

    if share > MAX_GEOMETRY_DISAGREEMENT:
        raise ValueError(
            f"screen geometry disagrees with the stored token counts on "
            f"{len(corrected)} of {n_trials} trials ({share:.1%}). That is far too "
            "many to be stimulus-text defects, so the AREA_BAND_* boundaries "
            "almost certainly do not describe this dataset's display. Establish "
            "the layout before trusting either assignment."
        )

    if len(corrected):
        shown = ", ".join(f"{pid} trial {tid}" for tid, pid in corrected[:10])
        more = "" if len(corrected) <= 10 else f", +{len(corrected) - 10} more"
        print(
            f"  screen areas: on-screen rectangles overrode the stored token counts "
            f"for {int(disagree.sum())} interest area(s) across {len(corrected)} of "
            f"{n_trials} trial(s) -- {shown}{more}"
        )
    else:
        print(
            f"  screen areas: token counts and on-screen rectangles agree on all "
            f"{n_trials} trials"
        )

    df[C.AREA_SCREEN_LOCATION] = by_geometry
    return df
