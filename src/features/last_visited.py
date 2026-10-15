"""Which area the participant looked at last -- three versions.

    last_lbl_before_select        the area looked at just before the final
                                  selection press
    last_lbl_before_confirm       the area looked at just before the
                                  confirming press
    last_answer_area_visited_lbl  the last ANSWER looked at in the trial
                                  (ignoring the question)

The first two take the press times from the button-press table
(`button_clicks_data.csv`, see `ingest/clicks.py`) and the fixations from the
fixation report, placed on words by `fixations.place_fixations`. The third comes
from the word table alone, with no button times. They are built differently, so
they need not agree.
"""

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ast
import pandas as pd

from src.config import columns as C
from src.config.datasets import BUTTON_CLICKS_PATH
from src.features.fixations import place_fixations


def compute_last_area_labels(
    ia_df: pd.DataFrame,
    fixations: pd.DataFrame,
    trial_level_path: Path = BUTTON_CLICKS_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """The word and area looked at just before the last selection press, and
    just before the confirming press. This is what the pipeline uses.

    The press times come from the button-press table at `trial_level_path`.
    `fixations` is the answer screen's fixation table
    (`fixations.load_fixations`); each fixation is placed on a word with
    `fixations.place_fixations`. For each press, the fixation is the last one
    that started at or before it. The area is missing if the trial has no such
    press, no fixation before it, or only a fixation that could not be placed
    on a word.

    Returns one row per trial that is both in `ia_df` and in the press table
    (which also holds practice and excluded trials), in the press table's
    order, with the word and the area for each press (`C.LAST_IA_BEFORE_SELECT`,
    `C.AREA_LABEL_BEFORE_SELECT`, `C.LAST_IA_BEFORE_CONFIRM`,
    `C.AREA_LABEL_BEFORE_CONFIRM`).
    """
    keys = [C.PARTICIPANT_ID, C.TRIAL_ID]
    if verbose:
        print("Finding the fixation before each button press...")

    presses = pd.read_csv(
        trial_level_path, usecols=keys + [C.SELECT_ANS_TIMESTAMPS, C.CONFIRM_TIMESTAMPS]
    )
    out = presses[keys].merge(ia_df[keys].drop_duplicates(), on=keys, how="inner")

    placed = place_fixations(
        fixations, ia_df, [C.AREA_LABEL_COLUMN], label="last look", verbose=verbose
    )
    placed = placed[keys + [C.CURRENT_FIX_START, C.INTEREST_AREA_ID, C.AREA_LABEL_COLUMN]]
    placed = placed.rename(columns={C.CURRENT_FIX_START: "_time"})
    placed["_time"] = placed["_time"].astype("float64")
    placed = placed.sort_values("_time", kind="mergesort")

    for press_col, ia_col, area_col in (
        (C.SELECT_ANS_TIMESTAMPS, C.LAST_IA_BEFORE_SELECT, C.AREA_LABEL_BEFORE_SELECT),
        (C.CONFIRM_TIMESTAMPS, C.LAST_IA_BEFORE_CONFIRM, C.AREA_LABEL_BEFORE_CONFIRM),
    ):
        # The trial's last press of this kind. The press table stores each
        # trial's press times as a text list, e.g. "[45120, 47380]".
        last_press = presses[keys].copy()
        last_press["_time"] = presses[press_col].map(
            lambda s: max(ast.literal_eval(s), default=None) if isinstance(s, str) else None
        ).astype("float64")
        last_press = last_press.dropna(subset=["_time"]).sort_values("_time", kind="mergesort")

        found = pd.merge_asof(
            last_press, placed, on="_time", by=keys, direction="backward"
        ).rename(columns={C.INTEREST_AREA_ID: ia_col, C.AREA_LABEL_COLUMN: area_col})
        out = out.merge(found[keys + [ia_col, area_col]], on=keys, how="left")
        out[ia_col] = out[ia_col].astype("Int64")

    return out[
        keys
        + [
            C.LAST_IA_BEFORE_SELECT,
            C.AREA_LABEL_BEFORE_SELECT,
            C.LAST_IA_BEFORE_CONFIRM,
            C.AREA_LABEL_BEFORE_CONFIRM,
        ]
    ]


# ---------------------------------------------------------------------------
# Registry entry: the last answer looked at, from the word table alone
# ---------------------------------------------------------------------------


def create_last_area_and_location_visited(df: pd.DataFrame) -> pd.DataFrame:
    """The last ANSWER looked at in each trial, by meaning and by place.

    Among the trial's answer words, takes the one whose last fixation started
    latest. The question is ignored, so if the very last look was at the
    question, this is the last answer before it.

    Careful: a word nobody looked at has last-fixation time "." and counts as
    0. So in a trial where no answer word was looked at, all are tied at 0 and
    one is picked arbitrarily -- the trial still gets an answer it never looked
    at. This happens on 28 Study 1 trials and 1 Study 2 trial.
    """
    df_local = df.copy()

    df_local[C.IA_LAST_FIXATION_TIME] = (
        df_local[C.IA_LAST_FIXATION_TIME].replace(".", 0).astype(int)
    )

    df_sorted = df_local.sort_values(
        by=[C.TRIAL_ID, C.PARTICIPANT_ID, C.IA_LAST_FIXATION_TIME],
        ascending=[True, True, False],
    )

    df_answers_only = df_sorted[df_sorted[C.AREA_LABEL_COLUMN].isin(C.ANSWER_AREAS)]

    last_fix = df_answers_only.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID], as_index=False
    ).head(1)

    result = last_fix[
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN, C.AREA_SCREEN_LOCATION]
    ].rename(
        columns={
            C.AREA_LABEL_COLUMN: C.LAST_VISITED_LABEL,
            C.AREA_SCREEN_LOCATION: C.LAST_VISITED_LOCATION,
        }
    )

    return result.reset_index(drop=True)
