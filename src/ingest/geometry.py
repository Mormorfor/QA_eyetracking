"""Which screen area a word sits in, from its on-screen rectangle.

This is the T3.18 fix. Area labels used to be derived by counting tokens
against the stored stimulus text, which silently shifted every boundary after
a displaced quote -- 20 L1 trials were affected and nobody had looked. Labels
now come from IA_TOP / IA_LEFT, with the token-count assignment kept as a
cross-check: geometry wins where they disagree, every override is printed, and
a disagreement rate above MAX_GEOMETRY_DISAGREEMENT aborts the run rather than
quietly relabelling a dataset whose screen template does not match.

It lives in ingest/ because it encodes the *screen layout* -- a property of how
the experiment was displayed, like the report format, not a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import columns as C


# ---------------------------------------------------------------------------
#  Answer-screen layout
# ---------------------------------------------------------------------------
# The answer screen is a fixed template: the question on one or two lines at the
# top, then the four options in a diamond -- one above, two side by side, one
# below. Every interest-area rectangle therefore falls into one of four vertical
# bands, and the middle band splits by x into the left and right options.
#
# The numbers are midpoints of the gaps measured on OneStop's L1 export: bands at
# IA_TOP 153-265 / 381-720 / 723-1062 / 1065-1294, with the left option ending at
# IA_LEFT 873 and the right one starting at 1717. Across 759,990 interest areas
# no IA_TOP value falls in two bands and no IA_LEFT value in both side options.
# Study 2 reuses the same screen (KnowQA's IA_TOP spans 153-1290), so the same
# template applies.
#
# NB: these bands cannot be recovered from the gaps within a single trial. Line
# spacing inside one area is ~110 px while the gap between the top band and the
# middle one is 3 px, so gap-clustering finds line breaks, not area boundaries.
# The template has to be stated, and is cross-checked on every run below.
AREA_BAND_QUESTION_TOP = 323.0


AREA_BAND_TOP_MIDDLE = 721.5


AREA_BAND_MIDDLE_BOTTOM = 1063.5


AREA_SPLIT_LEFT_RIGHT = 1295.0


# If the template ever stops describing a dataset's display, geometry and token
# counts disagree on essentially every trial rather than on a handful. Refuse
# rather than relabel a whole run from a template that does not fit it.
MAX_GEOMETRY_DISAGREEMENT = 0.05


def assign_area_by_geometry(df: pd.DataFrame) -> np.ndarray:
    """Assign each interest area to a screen area from its on-screen rectangle.

    Unlike the token-count assignment this reads the measurement rather than an
    inference from stored text, so it is unaffected by a stored question that
    carries a token the screen never rendered, or by options reaching the report
    in a different order than the template assumes.
    """
    missing = [c for c in ("IA_TOP", "IA_LEFT") if c not in df.columns]
    if missing:
        raise KeyError(
            f"interest-area geometry {missing} is required to place words on the "
            "answer screen; it is present in every raw report this project reads"
        )

    top = pd.to_numeric(df["IA_TOP"]).to_numpy()
    left = pd.to_numeric(df["IA_LEFT"]).to_numpy()
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
    """Cross-check the token-count assignment against the on-screen rectangles.

    The two agree on all but a handful of trials. Where they differ the rectangle
    wins: it is what was measured, while the token count is an inference from
    stored text the display may not have rendered word for word. Corrections are
    printed rather than applied quietly. See `docs/todo.md` T3.18 for the three
    ways the stored text drifts from the display and what each one costs.
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
