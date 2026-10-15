"""Which area the participant looked at last -- three versions.

    last_lbl_before_select        the area looked at just before the final
                                  selection press
    last_lbl_before_confirm       the area looked at just before the
                                  confirming press
    last_answer_area_visited_lbl  the last ANSWER looked at in the trial
                                  (ignoring the question)

The first two are timed from the button presses (`button_clicks_data.csv`, see
`ingest/clicks.py`). The third comes from the word table alone, with no button
times. They are built differently, so they need not agree.
"""

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ast
import pandas as pd

from src.config import columns as C
from src.config.datasets import BUTTON_CLICKS_PATH, GATHERERS_LAST_PATH, GATHERERS_PROCESSED_PATH, HUNTERS_LAST_PATH, HUNTERS_PROCESSED_PATH


def _parse_tuple_list(x):
    """Read a stored list of (time, word id) pairs; empty if missing."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)


def _extract_last_ia_from_tuple_list(x):
    """The word id of the last (time, word id) pair in a list.

    [(42231, 24), (44993, 9), (45873, 45), (46380, 32)] -> 32

    Returns a missing value (no error) for an empty list or anything that
    cannot be read.
    """

    try:
        tuples_ = _parse_tuple_list(x)
        if not tuples_:
            return pd.NA

        last_item = tuples_[-1]

        if isinstance(last_item, tuple) and len(last_item) == 2:
            ia = last_item[1]
            if pd.isna(ia):
                return pd.NA
            return int(ia)

        return pd.NA
    except Exception:
        return pd.NA


def extract_last_areas_from_trial_level_df(
    trial_df: pd.DataFrame,
    participant_col: str = "participant_id",
    trial_col: str = "TRIAL_INDEX",
    select_fix_col: str = "LAST_FIXATIONS_BEFORE_SELECT",
    confirm_fix_col: str = "LAST_FIXATIONS_BEFORE_CONFIRM",
) -> pd.DataFrame:
    """From the button-press table, the word looked at just before the last
    selection press and just before the confirming press. One row per trial."""
    out = trial_df[[participant_col, trial_col, select_fix_col, confirm_fix_col]].copy()

    out["last_ia_before_last_select"] = out[select_fix_col].apply(
        _extract_last_ia_from_tuple_list
    )

    out["last_ia_before_confirm"] = out[confirm_fix_col].apply(
        _extract_last_ia_from_tuple_list
    )

    return out[
        [
            participant_col,
            trial_col,
            "last_ia_before_last_select",
            "last_ia_before_confirm",
        ]
    ]


def attach_area_labels(
    extracted_df: pd.DataFrame,
    ia_df: pd.DataFrame,
    participant_col: str = "participant_id",
    trial_col: str = "TRIAL_INDEX",
    ia_id_col: str = "IA_ID",
    area_label_col: str = "area_label",
) -> pd.DataFrame:
    """Turn the two word ids into areas, by looking them up in the word table.

    Only trials that are in `ia_df` are kept: the press table also holds
    practice and excluded trials. A word id that is missing, or not one of the
    trial's words, gives a missing area.
    """

    ia_small = ia_df[[participant_col, trial_col, ia_id_col, area_label_col]].copy()

    ia_small[ia_id_col] = pd.to_numeric(ia_small[ia_id_col], errors="coerce").astype(
        "Int64"
    )

    valid_pairs = ia_small[[participant_col, trial_col]].drop_duplicates()

    extracted = extracted_df.copy()
    extracted["last_ia_before_last_select"] = pd.to_numeric(
        extracted["last_ia_before_last_select"], errors="coerce"
    ).astype("Int64")

    extracted["last_ia_before_confirm"] = pd.to_numeric(
        extracted["last_ia_before_confirm"], errors="coerce"
    ).astype("Int64")

    # keep only the trials that are in the word table
    extracted = extracted.merge(
        valid_pairs,
        on=[participant_col, trial_col],
        how="inner",
    )

    select_lookup = ia_small.rename(
        columns={
            ia_id_col: "last_ia_before_last_select",
            area_label_col: "area_label_before_last_select",
        }
    )

    confirm_lookup = ia_small.rename(
        columns={
            ia_id_col: "last_ia_before_confirm",
            area_label_col: "area_label_before_confirm",
        }
    )

    out = extracted.merge(
        select_lookup,
        on=[participant_col, trial_col, "last_ia_before_last_select"],
        how="left",
    )

    out = out.merge(
        confirm_lookup,
        on=[participant_col, trial_col, "last_ia_before_confirm"],
        how="left",
    )

    return out[
        [
            participant_col,
            trial_col,
            "last_ia_before_last_select",
            "area_label_before_last_select",
            "last_ia_before_confirm",
            "area_label_before_confirm",
        ]
    ]


def compute_last_area_labels(
    ia_df: pd.DataFrame,
    trial_level_path: Path = BUTTON_CLICKS_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """The area looked at just before the last selection and just before the
    confirmation, for every trial in `ia_df`. This is what the pipeline uses.

    Reads the button-press table at `trial_level_path`. Returns one row per
    trial: the word id and area before the last selection, and before the
    confirmation.
    """
    if verbose:
        print("Extracting last IAs from trial-level data...")
    trial_df = pd.read_csv(trial_level_path)
    extracted = extract_last_areas_from_trial_level_df(trial_df)

    if verbose:
        print("Attaching area labels...")
    return attach_area_labels(extracted, ia_df)


def get_last_areas_from_trial_level_csv(
    trial_level_path: Path = BUTTON_CLICKS_PATH,
    hunt_path: Path = HUNTERS_PROCESSED_PATH,
    gath_path: Path = GATHERERS_PROCESSED_PATH,
    out_hunt_path: Path = HUNTERS_LAST_PATH,
    out_gath_path: Path = GATHERERS_LAST_PATH,
    verbose=True,
):
    """The same two labels, built separately for hunters and gatherers and saved.

    Reads `hunters.csv` and `gatherers.csv`, which are only written on request
    (see `config/datasets.py`), so on a normal build this fails. The pipeline
    uses `compute_last_area_labels` instead.
    """
    if verbose:
        print("Loading data...")

    trial_df = pd.read_csv(trial_level_path)
    hunt = pd.read_csv(hunt_path)
    gath = pd.read_csv(gath_path)

    if verbose:
        print("Extracting last IAs from trial-level data...")

    extracted = extract_last_areas_from_trial_level_df(trial_df)

    if verbose:
        print("Attaching area labels (hunters)...")

    fin_hunt = attach_area_labels(extracted, hunt)

    if verbose:
        print("Attaching area labels (gatherers)...")

    fin_gath = attach_area_labels(extracted, gath)

    if verbose:
        print("Saving outputs...")

    fin_hunt.to_csv(out_hunt_path, index=False)
    fin_gath.to_csv(out_gath_path, index=False)

    if verbose:
        print(f"Saved:\n- {out_hunt_path}\n- {out_gath_path}")

    return fin_hunt, fin_gath


if __name__ == "__main__":
    get_last_areas_from_trial_level_csv()

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

    answer_labels = [lab for lab in C.LABEL_CHOICES if lab != "question"]
    df_answers_only = df_sorted[df_sorted[C.AREA_LABEL_COLUMN].isin(answer_labels)]

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
