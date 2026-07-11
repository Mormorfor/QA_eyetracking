import os
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import numpy as np
import ast
import os
import itertools
from collections import Counter

from src import constants as C
from src.data_paths import (
    ALL_PARTICIPANTS_LAST_PATH,
    ALL_PARTICIPANTS_PROCESSED_PATH,
    BUTTON_CLICKS_PATH,
    FIX_A_TSV_PATH,
    FIX_ANSWERS_PATH,
    GATHERERS_PROCESSED_PATH,
    HUNTERS_PROCESSED_PATH,
    IA_ANSWERS_PATH,
    IA_PARAGRAPH_PATH,
    PARTICIPANT_PUPILS_PATH,
    RT_AND_TFD_PATH,
)
from src.data_prep.button_clicks_processing import run_trial_level_pipeline
from src.derived.pupil_norm import (
    get_participant_pupil_stats,
    scale_pupil_area_to_mm,
    zscore_pupil_by_participant,
)
from src.derived.reading_times import build_rt_and_tfd
from src.derived.select_confirm_last import compute_last_area_labels

# ===========================================================================
# HOW TO ADD A NEW FEATURE FUNCTION
# ===========================================================================
# There are two kinds of feature functions:
#
# 1. Base per-row features
#    - operate on raw IA data
#    - return a full DataFrame with the same “grain” (one row per IA)
#
# 2. Group-level features
#    - aggregate per TRIAL/PARTICIPANT/(AREA (sometimes))
#    - return a *smaller* DataFrame that is later merged back on join columns
#
# Both kinds are registered in FUNCTION_REGISTRY (see below). If you do NOT
# pass base_function_names / group_function_names to main(), then:
#   - all "base" functions in FUNCTION_REGISTRY are run in registry order
#   - all "group" functions in FUNCTION_REGISTRY are run in registry order
#
# ---------------------------------------------------------------------------
# To add a NEW BASE FEATURE
# ---------------------------------------------------------------------------
# 1. (Optional) Add any NEW column name to constants.py, e.g.:
#       NEW_FEATURE_COLUMN = "my_new_feature"
#
# 2. Implement the function here with the signature:
#       def add_my_new_feature(df: pd.DataFrame) -> pd.DataFrame:
#           df = df.copy()
#           # ... compute feature ...
#           df[C.NEW_FEATURE_COLUMN] = ...
#           return df
#
# 3. Register it in FUNCTION_REGISTRY with kind="base":
#       "add_my_new_feature": {
#           "callable": add_my_new_feature,
#           "default_kwargs": {},
#           "kind": "base",
#       }
#
#   After this, it will automatically be included in the base pipeline
#   whenever main() is called without base_function_names.
#
# ---------------------------------------------------------------------------
# To add a NEW GROUP-LEVEL METRIC
# ---------------------------------------------------------------------------
# 1. Add a new metric constant to constants.py, e.g.:
#       NEW_METRIC = "my_metric"
#
# 2. (Optional, if it’s a standard area-level metric) add it to
#    AREA_METRIC_COLUMNS in constants.py:
#       AREA_METRIC_COLUMNS = [
#           ...,
#           NEW_METRIC,
#       ]
#
# 3. Implement the function here with signature:
#       def create_my_metric(df: pd.DataFrame) -> pd.DataFrame:
#           df = df.copy()
#           # ... any preprocessing ...
#           return (
#               df.groupby(
#                   [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN],
#                   as_index=False
#               ).agg(**{
#                   C.NEW_METRIC: (C.SOME_SOURCE_COLUMN, "mean")
#               })
#           )
#
#    The function must return a DataFrame that contains all join columns
#    plus the new metric column(s).
#
# 4. Register it in FUNCTION_REGISTRY with kind="group":
#       "create_my_metric": {
#           "callable": create_my_metric,
#           "default_kwargs": {
#               "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
#               # or any other set of columns you want to merge on
#           },
#           "kind": "group",
#       }
#
#   After this, it will automatically be included in the group pipeline
#   whenever main() is called without group_function_names.
#
# ---------------------------------------------------------------------------
# Overriding which functions run
# ---------------------------------------------------------------------------
# In main(), you can still override what runs:
#
#   main(
#       base_function_names=["add_text_id", "add_is_correct"],
#       group_function_names=[
#           "create_mean_area_dwell_time",
#           ("create_my_metric", {"join_columns": [C.TRIAL_ID]})
#       ],
#   )
#
# - For base_function_names, pass a list of function names (strings).
# - For group_function_names, each item can be:
#     * "func_name"
#     * ("func_name", {override_kwargs})
#
# Any unknown name will raise a ValueError.
# ===========================================================================


# ---------------------------------------------------------------------------
# Raw Data Loading
# ---------------------------------------------------------------------------


def load_raw_answers_data(ia_a_path: Path = IA_ANSWERS_PATH):
    """
    Load raw interest area level answers data from CSV file.
    """
    return pd.read_csv(ia_a_path, engine="python")


def load_raw_paragraphs_data(ia_p_path: Path = IA_PARAGRAPH_PATH):
    """
    Load raw interest area level paragraphs data from CSV file.
    """
    return pd.read_csv(ia_p_path, engine="python")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


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


def add_text_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a unique text identifier column combining article, difficulty, batch, and paragraph.
    """
    out = df.copy()
    out[C.TEXT_ID_COLUMN] = (
        out[C.BATCH_COLUMN].astype(str)
        + "_"
        + out[C.ARTICLE_COLUMN].astype(str)
        + "_"
        + out[C.DIFFICULTY_COLUMN].astype(str)
        + "_"
        + out[C.PARAGRAPH_COLUMN].astype(str)
    )
    return out


def add_text_id_with_q(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'text_id_with_q' column that matches the original answer-text logic:
    text_id_with_q = text_id + '_' + same_critical_span
    """
    df = df.copy()

    df[C.TEXT_ID_WITH_Q_COLUMN] = (
        df[C.TEXT_ID_COLUMN].astype(str)
        + "_"
        + df[C.SAME_CRITICAL_SPAN_COLUMN].astype(str)
    )

    return df


def add_is_correct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds IS_CORRECT_COLUMN to the dataframe based on comparison of selected and correct answer positions.

    """
    out = df.copy()
    out[C.IS_CORRECT_COLUMN] = (
        out[C.SELECTED_ANSWER_POSITION_COLUMN] == out[C.CORRECT_ANSWER_POSITION_COLUMN]
    ).astype(int)
    return out


def add_answer_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates explicit answer text columns (answer_A, answer_B, answer_C, answer_D)
    per answer label (correctness level).

    based on the answer order and the screen location based
    answer_1, answer_2, answer_3, answer_4 columns.

    """
    df_out = df.copy()

    def get_answer_by_label(row, label):
        order = ast.literal_eval(row[C.ANSWERS_ORDER_COLUMN])
        answer_idx = order.index(label)
        return row[f"{C.ANSWER_PREFIX}{answer_idx + 1}"]

    for label in C.ANSWER_LABELS:
        df_out[f"answer_{label}"] = df_out.apply(
            lambda row, lab=label: get_answer_by_label(row, lab),
            axis=1,
        )
    return df_out


def add_IA_screen_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a screen-location label to each interest area within a trial.

    For each trial (TRIAL_ID, PARTICIPANT_ID), the function:
    - tokenizes question and answer_1–answer_4 text,
    - computes token lengths,
    - treats INTEREST_AREA_ID (1-based) as the token index,
    - assigns each IA to one of AREA_LABEL_CHOICES
    (ordered: question, answer on top, answer to the left, answer to the right, answer on bottom)

    """
    df = df.copy()
    for col in ["question", "answer_1", "answer_2", "answer_3", "answer_4"]:
        df[col] = df[col].fillna("").astype(str)

    df["question_tokens"] = df["question"].str.split()
    df["1_tokens"] = df["answer_1"].str.split()
    df["2_tokens"] = df["answer_2"].str.split()
    df["3_tokens"] = df["answer_3"].str.split()
    df["4_tokens"] = df["answer_4"].str.split()

    df["question_len"] = df["question_tokens"].apply(len)
    df["1_len"] = df["1_tokens"].apply(len)
    df["2_len"] = df["2_tokens"].apply(len)
    df["3_len"] = df["3_tokens"].apply(len)
    df["4_len"] = df["4_tokens"].apply(len)

    def assign_area(group):
        q_len = group["question_len"].iloc[0]
        first_len = group["1_len"].iloc[0]
        second_len = group["2_len"].iloc[0]
        third_len = group["3_len"].iloc[0]
        fourth_len = group["4_len"].iloc[0]

        q_end = q_len - 1
        first_end = q_len + first_len - 1
        second_end = q_len + first_len + second_len - 1
        third_end = q_len + first_len + second_len + third_len - 1
        fourth_end = q_len + first_len + second_len + third_len + fourth_len - 1

        index_id = group[C.INTEREST_AREA_ID] - 1

        conditions = [
            (index_id <= q_end),
            (index_id > q_end) & (index_id <= first_end),
            (index_id > first_end) & (index_id <= second_end),
            (index_id > second_end) & (index_id <= third_end),
            (index_id > third_end) & (index_id <= fourth_end),
        ]

        choices = C.LOC_CHOICES
        group[C.AREA_SCREEN_LOCATION] = np.select(
            conditions, choices, default="unknown"
        )
        return group

    df_area_split = (
        df.set_index([C.TRIAL_ID, C.PARTICIPANT_ID])
        .groupby([C.TRIAL_ID, C.PARTICIPANT_ID], group_keys=False)
        .apply(assign_area)
    )
    return df_area_split


def add_IA_answer_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a logical answer label (correctness level) per interest area based on its screen location and
    the trial-specific answers order.

    - If AREA_SCREEN_LOCATION == LOC_CHOICES[0], return 'question'.
    - Else, find position index p = LOC_CHOICES.index(loc) - 1 (0..3),
     take letter = answers_order[p] (A/B/C/D),
     and map it to 'answer_A' / 'answer_B' / 'answer_C' / 'answer_D'.

    """
    df_out = df.copy()

    letter_to_label = {
        "A": "answer_A",
        "B": "answer_B",
        "C": "answer_C",
        "D": "answer_D",
    }

    def get_area_label(row):
        loc = row[C.AREA_SCREEN_LOCATION]

        if loc == C.LOC_CHOICES[0]:
            return "question"

        if loc in C.LOC_CHOICES[1:]:
            # position index: 0..3 for answers
            pos_index = C.LOC_CHOICES.index(loc) - 1
            answers_order = ast.literal_eval(row[C.ANSWERS_ORDER_COLUMN])
            letter = answers_order[pos_index]
            return letter_to_label.get(letter, None)

        return None

    df_out[C.AREA_LABEL_COLUMN] = df_out.apply(get_area_label, axis=1)
    return df_out


def add_selected_answer_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a selected-answer-label (A/B/C/D) column based on the answer position
    and the trial-specific answer order.

    This converts a *location-based* selected answer (e.g. "selected position = 1")
    into a *label-based* answer (e.g. "selected answer = 'B'") by using the
    answers_order sequence stored per trial.

    """
    df = df.copy()
    df[C.ANSWERS_ORDER_COLUMN] = df[C.ANSWERS_ORDER_COLUMN].apply(ast.literal_eval)
    df[C.SELECTED_ANSWER_LABEL_COLUMN] = df.apply(
        lambda row: row[C.ANSWERS_ORDER_COLUMN][row[C.SELECTED_ANSWER_POSITION_COLUMN]],
        axis=1,
    )
    return df


def add_zscored_pupil_columns(
    df: pd.DataFrame,
    pupil_stats=None,
) -> pd.DataFrame:
    """
    1) Convert IA pupil columns to mm (stored back into original columns)
    2) Z-score them using participant stats
    3) Store z-scored values into new <column>_z columns

    `pupil_stats` is resolved via get_participant_pupil_stats() and may be a
    ready statistics DataFrame, or None to fall back to that resolver's
    defaults (compute on the fly from the raw fixation data).
    """
    pupil_stats = get_participant_pupil_stats(stats=pupil_stats)

    out = df.copy()
    out = out.reset_index()

    pupil_cols = [
        C.IA_MAX_FIX_PUPIL_SIZE,
        C.IA_MIN_FIX_PUPIL_SIZE,
        C.IA_AVERAGE_FIX_PUPIL_SIZE,
    ]

    for col in pupil_cols:
        out[col] = scale_pupil_area_to_mm(out[col])

        out = zscore_pupil_by_participant(
            df=out,
            pupil_col=col,
            participant_col=C.PARTICIPANT_ID,
            stats=pupil_stats,
            out_col=f"{col}_z",
        )

    return out


def add_total_answering_RT_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total_answering_RT_normalized by dividing total_answering_RT
    by the total number of words on the answer screen:
    question_len + 1_len + 2_len + 3_len + 4_len.
    """
    out = df.copy()

    len_cols = ["question_len", "1_len", "2_len", "3_len", "4_len"]

    out["total_words_on_screen"] = out[len_cols].sum(axis=1)

    out[C.TOTAL_ANSWERING_RT_NORMALIZED] = pd.to_numeric(
        out[C.CONFIRM_FINAL_ANSWER_RT], errors="coerce"
    ) / out["total_words_on_screen"].replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
#  Group Features Creation
# ---------------------------------------------------------------------------


def create_mean_area_dwell_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean dwell time per (trial, participant, area_label) group.

    This function aggregates interest-area-level data into area-level summaries
    by computing the mean dwell time for each unique combination of:
    - TRIAL_ID
    - PARTICIPANT_ID
    - AREA_LABEL_COLUMN (e.g., 'question', 'answer_A', ...)

    """
    df[C.IA_DWELL_TIME] = df[C.IA_DWELL_TIME].replace(".", 0).astype(int)
    return df.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False
    ).agg(**{C.MEAN_DWELL_TIME: (C.IA_DWELL_TIME, "mean")})


def create_mean_area_fix_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean number of fixations per (trial, participant, area_label) group.

    This function aggregates interest-area–level data by computing the mean
    number of fixations for each unique combination of:
    - TRIAL_ID
    - PARTICIPANT_ID
    - AREA_LABEL_COLUMN (e.g., 'question', 'answer_A', ...)

    """
    df[C.IA_FIXATIONS_COUNT] = df[C.IA_FIXATIONS_COUNT].replace(".", 0).astype(int)
    return df.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False
    ).agg(**{C.MEAN_FIXATIONS_COUNT: (C.IA_FIXATIONS_COUNT, "mean")})


def create_mean_first_fix_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean first-fixation duration per (trial, participant, area_label).

    This function:
    Groups by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN).
    Computes the mean first-fixation duration within each group.

    """
    df[C.IA_FIRST_FIXATION_DURATION] = (
        df[C.IA_FIRST_FIXATION_DURATION].replace(".", 0).astype(int)
    )
    return df.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False
    ).agg(**{C.MEAN_FIRST_FIXATION_DURATION: (C.IA_FIRST_FIXATION_DURATION, "mean")})


def create_skip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the skip rate per (trial, participant, area_label).

    A skip is defined as an interest area (IA) with *zero dwell time*.
    The skip rate is the proportion of IAs within an area (e.g., 'answer_A')
    that were skipped by the participant during the trial.

    - Create an indicator AREA_SKIPPED:
          1 if IA_DWELL_TIME == 0
          0 otherwise
    - Group by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN)
    - Compute the mean of AREA_SKIPPED → skip_rate

    """
    df[C.IA_DWELL_TIME] = df[C.IA_DWELL_TIME].replace(".", 0).astype(int)
    df[C.AREA_SKIPPED] = (df[C.IA_DWELL_TIME] == 0).astype(int)
    return df.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False
    ).agg(**{C.SKIP_RATE: (C.AREA_SKIPPED, "mean")})


def create_dwell_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dwell time proportions per area within each trial and participant.

    For each (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN), this function:
    1. Sums IA_DWELL_TIME to obtain TOTAL_IA_DWELL_TIME per area.
    2. Sums TOTAL_IA_DWELL_TIME over all areas within a trial/participant to
       obtain TOTAL_TRIAL_DWELL_TIME.
    3. Computes AREA_DWELL_PROPORTION as:
           TOTAL_IA_DWELL_TIME / TOTAL_TRIAL_DWELL_TIME

    Any resulting NaN values (e.g., if TOTAL_TRIAL_DWELL_TIME is 0) are replaced by 0.
    """
    df[C.IA_DWELL_TIME] = df[C.IA_DWELL_TIME].replace(".", 0).astype(int)
    aggregated_df = (
        df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False)
        .agg({C.IA_DWELL_TIME: "sum"})
        .rename(columns={C.IA_DWELL_TIME: C.TOTAL_IA_DWELL_TIME})
    )
    aggregated_df[C.TOTAL_TRIAL_DWELL_TIME] = aggregated_df.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID]
    )[C.TOTAL_IA_DWELL_TIME].transform("sum")
    aggregated_df[C.AREA_DWELL_PROPORTION] = (
        aggregated_df[C.TOTAL_IA_DWELL_TIME] / aggregated_df[C.TOTAL_TRIAL_DWELL_TIME]
    )
    aggregated_df = aggregated_df.fillna(0)

    return aggregated_df


def create_mean_pupil_size_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy()

    mm_cols = [
        C.IA_MAX_FIX_PUPIL_SIZE,
        C.IA_MIN_FIX_PUPIL_SIZE,
        C.IA_AVERAGE_FIX_PUPIL_SIZE,
    ]
    z_cols = [f"{c}_z" for c in mm_cols]

    for col in mm_cols + z_cols:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    agg_spec = {}

    agg_spec[C.MEAN_MAX_FIX_PUPIL_SIZE] = (C.IA_MAX_FIX_PUPIL_SIZE, "mean")
    agg_spec[C.MEAN_MIN_FIX_PUPIL_SIZE] = (C.IA_MIN_FIX_PUPIL_SIZE, "mean")
    agg_spec[C.MEAN_AVG_FIX_PUPIL_SIZE] = (C.IA_AVERAGE_FIX_PUPIL_SIZE, "mean")

    agg_spec[C.MEAN_MAX_FIX_PUPIL_SIZE_Z] = (f"{C.IA_MAX_FIX_PUPIL_SIZE}_z", "mean")
    agg_spec[C.MEAN_MIN_FIX_PUPIL_SIZE_Z] = (f"{C.IA_MIN_FIX_PUPIL_SIZE}_z", "mean")
    agg_spec[C.MEAN_AVG_FIX_PUPIL_SIZE_Z] = (f"{C.IA_AVERAGE_FIX_PUPIL_SIZE}_z", "mean")

    return df_local.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False
    ).agg(**agg_spec)


def create_first_encounter_pupil_size(df: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy()

    mm_col = C.IA_AVERAGE_FIX_PUPIL_SIZE
    z_col = f"{mm_col}_z"

    df_local[mm_col] = pd.to_numeric(df_local[mm_col], errors="coerce")
    df_local[z_col] = pd.to_numeric(df_local[z_col], errors="coerce")

    df_local = df_local[df_local[C.IA_FIRST_FIXATION_DURATION] > 0]

    df_local = df_local.sort_values(
        by=[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
    )

    first_fix = df_local.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False
    ).head(1)

    out_cols = [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
    rename_map = {}

    out_cols.append(mm_col)
    rename_map[mm_col] = C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE

    out_cols.append(z_col)
    rename_map[z_col] = C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z

    return first_fix[out_cols].rename(columns=rename_map)


def create_last_area_and_location_visited(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the last-visited *answer* area label and screen location per trial,
    using the most recent fixation on one of the answer areas.

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


def create_fixation_sequence_tags(df, fix_path: Path = FIX_ANSWERS_PATH):
    """
    Build fixation sequences per trial/participant in terms of area labels and locations.

    For each (TRIAL_ID, PARTICIPANT_ID), this function:
    - Reads the precomputed fixation sequence stored in INTEREST_AREA_FIXATION_SEQUENCE,
      assumed to be a serialized list of IA_IDs.
    - Maps known IA_IDs directly from the IA-level dataframe (`df`).
    - For unknown IA_IDs (i.e. IDs not present in the current group's IA_ID set),
      falls back to the next value from `fixations_df[C.NEAREST_IA]` among fixation rows
      whose CURRENT_FIX_INTEREST_AREAS is an empty list.
    - Maps the resolved IA_IDs to:
        * AREA_LABEL_COLUMN
        * AREA_SCREEN_LOCATION
    - Produces:
        * FIX_SEQUENCE_BY_LABEL
        * FIX_SEQUENCE_BY_LOCATION

    The first fixation is removed only if:
    - there are at least 2 labels
    - the first is "question"
    - the second is not "question"

    Assumptions
    -----------
    - `fixations_df` contains one row per fixation.
    - `fixations_df[C.CURRENT_FIX_INTEREST_AREAS]` contains either real Python lists
      or strings representing lists like "[]" or "[21]".
    - Rows with empty CURRENT_FIX_INTEREST_AREAS correspond in order to unknown entries
      in INTEREST_AREA_FIXATION_SEQUENCE for the same participant-trial.
    - `C.NEAREST_IA = 'CURRENT_FIX_NEAREST_INTEREST_AREA'`
    """

    fixations_df = (
        fix_path if isinstance(fix_path, pd.DataFrame) else pd.read_csv(fix_path)
    )
    result = []

    group_cols = [C.TRIAL_ID, C.PARTICIPANT_ID]

    for (trial_index, participant_id), group in df.groupby(group_cols):
        group_ids = set(group[C.INTEREST_AREA_ID].unique())

        id_to_label = dict(zip(group[C.INTEREST_AREA_ID], group[C.AREA_LABEL_COLUMN]))
        id_to_location = dict(
            zip(group[C.INTEREST_AREA_ID], group[C.AREA_SCREEN_LOCATION])
        )

        # -----------------------------
        # original serialized sequence
        # -----------------------------
        sequence_str = group[C.INTEREST_AREA_FIXATION_SEQUENCE].iloc[0]
        sequence = (
            ast.literal_eval(sequence_str)
            if isinstance(sequence_str, str)
            else sequence_str
        )

        # ---------------------------------------------------------
        # fixation-level fallback queue for unknown sequence entries
        # ---------------------------------------------------------
        fix_group = fixations_df[
            (fixations_df[C.TRIAL_ID] == trial_index)
            & (fixations_df[C.PARTICIPANT_ID] == participant_id)
        ].copy()

        def _parse_list(x):
            if isinstance(x, list):
                return x
            if pd.isna(x):
                return []
            if isinstance(x, str):
                x = x.strip()
                if not x:
                    return []
                return ast.literal_eval(x)
            return []

        fix_group[C.CURRENT_FIX_INTEREST_AREAS] = fix_group[
            C.CURRENT_FIX_INTEREST_AREAS
        ].apply(_parse_list)

        fallback_nearest_ids = (
            fix_group.loc[
                fix_group[C.CURRENT_FIX_INTEREST_AREAS].apply(len) == 0, C.NEAREST_IA
            ]
            .astype(int)
            .tolist()
        )

        fallback_pointer = 0
        resolved_sequence = []

        # --------------------------------------------
        # resolve each entry: direct IA or fallback IA
        # --------------------------------------------
        for ia_id in sequence:
            if ia_id in group_ids:
                resolved_ia_id = ia_id
            else:
                if fallback_pointer < len(fallback_nearest_ids):
                    resolved_ia_id = fallback_nearest_ids[fallback_pointer]
                    fallback_pointer += 1
                else:
                    print(
                        "Warning: No fallback IA left for trial {}, participant {}".format(
                            trial_index, participant_id
                        )
                    )
                    continue

            resolved_sequence.append(resolved_ia_id)

        label_sequence = [id_to_label[ia_id] for ia_id in resolved_sequence]
        location_sequence = [id_to_location[ia_id] for ia_id in resolved_sequence]

        if (
            len(label_sequence) >= 2
            and label_sequence[0] == "question"
            and label_sequence[1] != "question"
        ):
            trimmed_labels = label_sequence[1:]
            trimmed_locations = location_sequence[1:]
        else:
            trimmed_labels = label_sequence
            trimmed_locations = location_sequence

        result.append(
            {
                C.TRIAL_ID: trial_index,
                C.PARTICIPANT_ID: participant_id,
                C.FIX_SEQUENCE_BY_LABEL: trimmed_labels,
                C.FIX_SEQUENCE_BY_LOCATION: trimmed_locations,
            }
        )

    return pd.DataFrame(result)


def create_simplified_fixation_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simplified fixation sequences by collapsing consecutive fixations
    on the same area into a single step.

    For each (TRIAL_ID, PARTICIPANT_ID), this function:
    1. Reads the fixation sequence from INTEREST_AREA_FIXATION_SEQUENCE
       (a serialized list of IA_IDs, e.g. "[1, 2, 2, 3, 3, 3, 2]").
    2. Maps each IA_ID to:
       - AREA_LABEL_COLUMN       (e.g. 'question', 'answer_A', ...)
       - AREA_SCREEN_LOCATION    (e.g. 'top', 'left', ...)
    3. Filters out IA_IDs that are not present in the group's IA set.
    4. Collapses consecutive fixations on the same label into a single entry
       (run-length compression).

    Example
    -------
    Raw label sequence:
        ['question', 'question', 'answer_A', 'answer_A', 'answer_B']
    Simplified label sequence:
        ['question', 'answer_A', 'answer_B']

    The location sequence is compressed in parallel, taking the location
    of the first fixation in each run.

    """
    rows = []
    for (trial_id, participant_id), g in df.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID], sort=False
    ):
        r = g.iloc[0]

        labels = list(r[C.FIX_SEQUENCE_BY_LABEL] or [])
        locs = list(r[C.FIX_SEQUENCE_BY_LOCATION] or [])

        n = len(labels)

        simpl_labels = []
        simpl_locs = []

        prev_label = None
        for lab, loc in zip(labels, locs):
            if lab != prev_label:
                simpl_labels.append(lab)
                simpl_locs.append(loc)
                prev_label = lab

        rows.append(
            {
                C.TRIAL_ID: trial_id,
                C.PARTICIPANT_ID: participant_id,
                C.SIMPLIFIED_FIX_SEQ_BY_LABEL: tuple(simpl_labels),
                C.SIMPLIFIED_FIX_SEQ_BY_LOCATION: tuple(simpl_locs),
            }
        )

    return pd.DataFrame(rows)


def create_simplified_visit_counts(df: pd.DataFrame) -> pd.DataFrame:
    """

    For each (trial, participant):
      - Count occurrences of each label in SIMPLIFIED_FIX_SEQ_BY_LABEL
      - Count occurrences of each location in SIMPLIFIED_FIX_SEQ_BY_LOCATION

    Then for each (trial, participant, area_label):
      - num_label_visits = count of that area_label in label sequence
      - num_loc_visits   = count of that area's screen location in location sequence
    """
    df_local = df.copy()

    counts_rows = []
    for (trial_id, participant_id), g in df_local.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID]
    ):
        labels_seq = g[C.SIMPLIFIED_FIX_SEQ_BY_LABEL].iloc[0]
        locs_seq = g[C.SIMPLIFIED_FIX_SEQ_BY_LOCATION].iloc[0]

        label_counts = Counter(labels_seq)
        loc_counts = Counter(locs_seq)

        counts_rows.append(
            {
                C.TRIAL_ID: trial_id,
                C.PARTICIPANT_ID: participant_id,
                "_label_counts": label_counts,
                "_loc_counts": loc_counts,
            }
        )

    counts_df = pd.DataFrame(counts_rows)
    df_local = df_local.merge(counts_df, on=[C.TRIAL_ID, C.PARTICIPANT_ID], how="left")

    df_local[C.NUM_LABEL_VISITS] = df_local.apply(
        lambda r: (
            int(r["_label_counts"].get(r[C.AREA_LABEL_COLUMN], 0))
            if isinstance(r["_label_counts"], Counter)
            else 0
        ),
        axis=1,
    )

    df_local[C.NUM_LOC_VISITS] = df_local.apply(
        lambda r: (
            int(r["_loc_counts"].get(r[C.AREA_SCREEN_LOCATION], 0))
            if isinstance(r["_loc_counts"], Counter)
            else 0
        ),
        axis=1,
    )

    out = df_local.groupby(
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False
    ).agg(
        **{
            C.NUM_LABEL_VISITS: (C.NUM_LABEL_VISITS, "max"),
            C.NUM_LOC_VISITS: (C.NUM_LOC_VISITS, "max"),
        }
    )

    return out


# ---------------------------------------------------------------------------
#  Processing Pipelines
# ---------------------------------------------------------------------------

FUNCTION_REGISTRY = {
    # Base features
    "add_text_id": {
        "callable": add_text_id,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_text_id_with_q": {
        "callable": add_text_id_with_q,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_is_correct": {
        "callable": add_is_correct,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_answer_text_columns": {
        "callable": add_answer_text_columns,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_IA_screen_location": {
        "callable": add_IA_screen_location,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_IA_answer_label": {
        "callable": add_IA_answer_label,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_selected_answer_label": {
        "callable": add_selected_answer_label,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_zscored_pupil_columns": {
        "callable": add_zscored_pupil_columns,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_total_answering_RT_normalized": {
        "callable": add_total_answering_RT_normalized,
        "default_kwargs": {},
        "kind": "base",
    },
    # Group-level functions
    "create_mean_area_dwell_time": {
        "callable": create_mean_area_dwell_time,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_mean_area_fix_count": {
        "callable": create_mean_area_fix_count,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_mean_first_fix_duration": {
        "callable": create_mean_first_fix_duration,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_skip_rate": {
        "callable": create_skip_rate,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_dwell_proportions": {
        "callable": create_dwell_proportions,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_mean_pupil_size_metrics": {
        "callable": create_mean_pupil_size_metrics,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_first_encounter_pupil_size": {
        "callable": create_first_encounter_pupil_size,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_last_area_and_location_visited": {
        "callable": create_last_area_and_location_visited,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "group",
    },
    ## heavy iternal data loading here. Might fix later, slow for now.
    "create_fixation_sequence_tags": {
        "callable": create_fixation_sequence_tags,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID],
            # Forwarded to the function; main() overrides it with its fixations_path.
            "fix_path": FIX_ANSWERS_PATH,
        },
        "kind": "group",
    },
    "create_simplified_fixation_tags": {
        "callable": create_simplified_fixation_tags,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "group",
    },
    "create_simplified_visit_counts": {
        "callable": create_simplified_visit_counts,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
}


def resolve_base_functions(name_list=None):
    """
    Resolve base feature functions.

    If name_list is None, return ALL base functions from FUNCTION_REGISTRY
    in registry insertion order. Otherwise, return only the named ones.

    can accept entries in name_list as either:
    - "func_name"
    - ("func_name", {override_kwargs})
    """
    # Case 1: no explicit list → use all base functions
    if name_list is None:
        return [
            (entry["callable"], entry.get("default_kwargs", {}))
            for name, entry in FUNCTION_REGISTRY.items()
            if entry.get("kind") == "base"
        ]

    # Case 2: explicit list → validate and return only those
    resolved = []
    for item in name_list:
        if isinstance(item, str):
            name = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown base function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "base":
                raise ValueError(
                    f"Function '{name}' is not registered as a base feature."
                )
            resolved.append((entry["callable"], entry.get("default_kwargs", {})))
            continue

        if isinstance(item, tuple) and len(item) == 2:
            name, user_kwargs = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown base function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "base":
                raise ValueError(
                    f"Function '{name}' is not registered as a base feature."
                )
            merged_kwargs = {**entry.get("default_kwargs", {}), **user_kwargs}
            resolved.append((entry["callable"], merged_kwargs))
            continue

        raise ValueError(
            f"Invalid base function specification: {item}. "
            "Must be 'name' or ('name', {kwargs})."
        )

    return resolved


def resolve_group_functions(name_list=None):
    """
    Resolve group-level feature functions.

    If name_list is None, return ALL group functions from FUNCTION_REGISTRY
    (with their default kwargs). Otherwise, name_list can contain:
        - "func_name"
        - ("func_name", {override_kwargs})
    """
    # Case 1: no explicit list → all group functions with their defaults
    if name_list is None:
        return [
            (entry["callable"], entry.get("default_kwargs", {}))
            for name, entry in FUNCTION_REGISTRY.items()
            if entry.get("kind") == "group"
        ]

    # Case 2: explicit list
    resolved = []
    for item in name_list:
        if isinstance(item, str):
            name = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "group":
                raise ValueError(
                    f"Function '{name}' is not registered as a group feature."
                )
            resolved.append((entry["callable"], entry.get("default_kwargs", {})))
            continue

        if isinstance(item, tuple) and len(item) == 2:
            name, user_kwargs = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "group":
                raise ValueError(
                    f"Function '{name}' is not registered as a group feature."
                )

            merged_kwargs = {**entry.get("default_kwargs", {}), **user_kwargs}
            resolved.append((entry["callable"], merged_kwargs))
            continue

        raise ValueError(
            f"Invalid group function specification: {item}. "
            "Must be 'name' or ('name', {kwargs})."
        )

    return resolved


def add_base_features(df, functions, verbose=False):
    """
    Apply a sequence of transformation functions to a DataFrame.

    Each function in `functions` must take a single DataFrame as input and
    return a (transformed) DataFrame as output. The functions are applied
    in the order given.
    """

    out = df.copy()
    for func, kwargs in functions:
        if verbose:
            print(f"Running: {func.__name__}")
        out = func(out, **kwargs) if kwargs else func(out)
    return out.reset_index(drop=False)


# Keys inside a group function's kwargs that configure the orchestrator (how the
# result is merged back) rather than arguments forwarded to the feature function.
GROUP_ORCHESTRATION_KEYS = {"join_columns"}


def generate_new_row_features(functions, df, default_join_columns=None, verbose=True):
    """
    Iteratively compute and merge group-level features into a row-level DataFrame.

    Each entry in `functions` is a tuple:
        (func, func_kwargs)

    `func_kwargs` is split into two roles:
    - "join_columns" (and any other GROUP_ORCHESTRATION_KEYS) are consumed here to
      control the left-merge and are NOT passed to `func`.
    - every remaining key is forwarded as a keyword argument to `func`, e.g. a
      `fix_path` that points the fixation-sequence feature at a specific fixations
      report.

    For each function:
    1. Compute `new_features_df = func(result_df, **forwarded_kwargs)`
    2. Merge `new_features_df` into `result_df` using a left join on `join_columns`.

    Returns
    -------
    DataFrame
        The original DataFrame enriched with all new feature columns produced
        by the functions in `functions`.
    """
    if default_join_columns is None:
        default_join_columns = [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]

    result_df = df.copy()

    for func, func_kwargs in functions:
        if verbose:
            print(f"Running group feature: {func.__name__}")

        join_columns = func_kwargs.get("join_columns", default_join_columns)
        call_kwargs = {
            key: value
            for key, value in func_kwargs.items()
            if key not in GROUP_ORCHESTRATION_KEYS
        }

        new_features_df = func(result_df, **call_kwargs)
        result_df = result_df.merge(new_features_df, on=join_columns, how="left")

    return result_df


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def _attach_last_label_features(
    df: pd.DataFrame,
    save_path: Path = None,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute last-area-label features directly from the processed IA data and
    merge in:
    - C.LAST_LBL_BEFORE_SELECT
    - C.LAST_LBL_BEFORE_CONFIRM

    The features are derived on the fly (via
    src.derived.select_confirm_last.compute_last_area_labels) from the trial-level
    button-click table at `button_clicks_path`. If `save_path` is given, the
    computed (unmerged) last-label table is written there as an auxiliary artifact.
    """
    if verbose:
        print("Computing last-area-label features…")

    last_df = compute_last_area_labels(
        df, trial_level_path=button_clicks_path, verbose=verbose
    )

    if save_path is not None:
        _save(last_df, save_path, label="last-area labels", verbose=verbose)

    merge_cols = [C.PARTICIPANT_ID, C.TRIAL_ID]
    rename_map = {
        "area_label_before_last_select": C.LAST_LBL_BEFORE_SELECT,
        "area_label_before_confirm": C.LAST_LBL_BEFORE_CONFIRM,
    }

    last_df = (
        last_df[merge_cols + list(rename_map.keys())]
        .drop_duplicates(subset=merge_cols)
        .rename(columns=rename_map)
    )

    return df.merge(last_df, on=merge_cols, how="left")


def _attach_rt_and_tfd_features(
    df: pd.DataFrame,
    save_path: Path = None,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    include_paragraph: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute RT_pure_/RT_normalized_/TFD_pure_/TFD_normalized_ features directly
    from the processed IA data and merge them in at trial level on
    (participant_id, TRIAL_INDEX).

    The features are derived on the fly (via
    src.derived.reading_times.build_rt_and_tfd), reading the trial's fixation
    sequence from `button_clicks_path` for the run-based RT. `include_paragraph`
    should be False for experiments without a paragraph-reading screen (then only
    answer-region RT/TFD is returned). If `save_path` is given, the computed RT/TFD
    table is written there as an auxiliary artifact.
    """
    if verbose:
        print("Computing RT/TFD features…")

    rt_df = build_rt_and_tfd(
        all_participants=df,
        button_clicks_path=button_clicks_path,
        include_paragraph=include_paragraph,
        save=save_path is not None,
        output_path=save_path,
        verbose=verbose,
    )

    merge_cols = [C.PARTICIPANT_ID, C.TRIAL_ID]
    rt_df = rt_df.drop_duplicates(subset=merge_cols)
    return df.merge(rt_df, on=merge_cols, how="left")


def _process(
    df: pd.DataFrame,
    base_funcs,
    group_funcs,
    add_last: bool = True,
    add_rts: bool = True,
    last_labels_path: Path = None,
    rt_and_tfd_path: Path = None,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    include_paragraph: bool = True,
    label: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run base + group pipelines on `df`, optionally attach last-label and
    RT/TFD features (computed on the fly), and return the enriched DataFrame.

    Both attached features read the trial-level button clicks from
    `button_clicks_path`. `include_paragraph` is forwarded to the RT/TFD step.
    `last_labels_path` / `rt_and_tfd_path`, when given, are where the computed
    auxiliary tables are saved."""
    if verbose:
        print(f"\nProcessing {label} (row-level)…")
    out = add_base_features(df, base_funcs, verbose=verbose)

    if verbose:
        print(f"Applying group-level features for {label}…")
    out = generate_new_row_features(group_funcs, out)

    if add_last:
        out = _attach_last_label_features(
            out,
            save_path=last_labels_path,
            button_clicks_path=button_clicks_path,
            verbose=verbose,
        )

    if add_rts:
        out = _attach_rt_and_tfd_features(
            out,
            save_path=rt_and_tfd_path,
            button_clicks_path=button_clicks_path,
            include_paragraph=include_paragraph,
            verbose=verbose,
        )

    return out


def _save(df: pd.DataFrame, output_path: Path, label: str = "", verbose: bool = True):
    """Write `df` to `output_path`, creating parent directories as needed."""
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    if verbose:
        print(f"Saving {label} features to: {output_path}")
    df.to_csv(output_path, index=False)


def _save_splits(
    df: pd.DataFrame,
    split_column: str,
    base_output_path: Path,
    split_output_paths: dict = None,
    verbose: bool = True,
):
    """Partition `df` by the unique values of `split_column` and save each subset.

    The target path for a value is taken from `split_output_paths` (a
    {value: path} mapping) when provided; otherwise it is derived from
    `base_output_path` as "<stem>_<split_column>_<value><suffix>".
    """
    if split_column not in df.columns:
        raise ValueError(
            f"split_column '{split_column}' not found in processed data."
        )

    base_output_path = Path(base_output_path)
    split_output_paths = split_output_paths or {}

    for value, subset in df.groupby(split_column):
        if value in split_output_paths:
            out_path = Path(split_output_paths[value])
        else:
            safe_value = str(value).replace(os.sep, "_").replace(" ", "_")
            out_path = base_output_path.with_name(
                f"{base_output_path.stem}_{split_column}_{safe_value}"
                f"{base_output_path.suffix}"
            )
        _save(subset, out_path, label=f"{split_column}={value}", verbose=verbose)


def main(
    ia_answers_path: Path = IA_ANSWERS_PATH,
    output_path: Path = ALL_PARTICIPANTS_PROCESSED_PATH,
    fixations_path: Path = FIX_ANSWERS_PATH,
    split_column: str = None,
    split_output_paths: dict = None,
    add_last: bool = True,
    add_rts: bool = True,
    include_paragraph: bool = True,
    compute_pupil_stats: bool = True,
    pupil_fixations_path: Path = None,
    pupil_stats_path: Path = PARTICIPANT_PUPILS_PATH,
    rebuild_button_clicks: bool = False,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    button_clicks_fix_csv_path: Path = FIX_ANSWERS_PATH,
    button_clicks_fix_tsv_path: Path = FIX_A_TSV_PATH,
    button_clicks_msg_participant_col: str = C.RECORDING_SESSION_LABEL,
    all_answers_is_cumulative: bool = True,
    save_auxiliary: bool = True,
    last_labels_path: Path = ALL_PARTICIPANTS_LAST_PATH,
    rt_and_tfd_path: Path = RT_AND_TFD_PATH,
    remove_repeats: bool = True,
    remove_practice: bool = True,
    base_function_names: list = None,
    group_function_names: list = None,
    verbose: bool = True,
):
    """
    Full preprocessing pipeline. Runs end-to-end from the raw DATA_RAW reports —
    no previously-generated file is required.

    By default, all trials are processed together (after filtering repeated and
    practice trials) and saved as a single combined CSV to `output_path`
    (all_participants.csv).

    `fixations_path` is the single canonical fixations report for the run. It feeds
    both the participant pupil stats and the fixation-sequence group feature
    (`create_fixation_sequence_tags`); point it at the matching fixations report when
    processing a different raw dataset.

    Intermediate ("auxiliary") artifacts are all derived on the fly and, when
    `save_auxiliary=True`, persisted under DATA/L1_based_data/Auxiliary:
    - participant pupil stats — computed from raw fixation data at
      `pupil_fixations_path` (which defaults to `fixations_path`) when
      `compute_pupil_stats=True`, else loaded from `pupil_stats_path`. Saved to
      `pupil_stats_path` when freshly computed.
    - button clicks — the one derived *input* the RT/last-label steps need, so it is
      only built when `add_last`/`add_rts` run. Rebuilt from raw (via
      button_clicks_processing.run_trial_level_pipeline) when `rebuild_button_clicks=True`
      or when `button_clicks_path` is missing, reading from `button_clicks_fix_csv_path`
      + `button_clicks_fix_tsv_path`. For a self-contained report that holds both the
      message and fixation columns (e.g. the new experiment fixations), pass
      `button_clicks_fix_tsv_path=None` and set `button_clicks_msg_participant_col`
      (e.g. participant_id) and `all_answers_is_cumulative=False` to match it.
    - last-area labels (`add_last`) → `last_labels_path`.
    - RT/TFD features (`add_rts`) → `rt_and_tfd_path`.

    If `split_column` is provided (e.g. C.QUESTION_PREVIEW_COLUMN to recover the
    hunters/gatherers split), the processed DataFrame is *additionally*
    partitioned by the unique values of that column and each partition is saved
    as its own CSV. Use `split_output_paths` (a {value: path} mapping) to control
    the per-split filenames, e.g.::

        main(
            split_column=C.QUESTION_PREVIEW_COLUMN,
            split_output_paths={
                True: HUNTERS_PROCESSED_PATH,
                False: GATHERERS_PROCESSED_PATH,
            },
        )
    """
    # The canonical fixations report feeds both the pupil stats and the
    # fixation-sequence group feature. pupil_fixations_path can still override just
    # the pupil-stats source when set explicitly; otherwise it follows fixations_path.
    if pupil_fixations_path is None:
        pupil_fixations_path = fixations_path

    # Button clicks is the one derived input the RT/last-label steps need, so only
    # build it when those steps run. Rebuild from raw when forced or absent, routing
    # to the configured fixations source (default: the legacy CSV + separate TSV;
    # pass button_clicks_fix_tsv_path=None for a self-contained new-data report).
    needs_button_clicks = add_last or add_rts
    if needs_button_clicks and (
        rebuild_button_clicks or not Path(button_clicks_path).exists()
    ):
        if verbose:
            reason = "forced" if rebuild_button_clicks else "missing"
            print(f"\nBuilding button-click data from raw ({reason})…")
        run_trial_level_pipeline(
            fix_csv_path=button_clicks_fix_csv_path,
            fix_tsv_path=button_clicks_fix_tsv_path,
            output_csv_path=Path(button_clicks_path),
            msg_participant_col=button_clicks_msg_participant_col,
            all_answers_is_cumulative=all_answers_is_cumulative,
            verbose=verbose,
        )

    if verbose:
        print(f"\nLoading raw answers from: {ia_answers_path}")

    df_answers = load_raw_answers_data(ia_answers_path)

    if remove_repeats:
        df_answers = df_answers[
            df_answers[C.REPEATED_TRIAL_COLUMN] == False
        ].copy()
    if remove_practice:
        df_answers = df_answers[
            df_answers[C.PRACTICE_TRIAL_COLUMN] == False
        ].copy()

    if verbose:
        print("\nResolving processing function lists…")

    base_funcs = resolve_base_functions(base_function_names)
    group_funcs = resolve_group_functions(group_function_names)

    # The fixations report has up to two consumers: the pupil-stats computation and
    # the fixation-sequence group feature. Load it once here and share the resulting
    # DataFrame with both, instead of letting each re-read the same large file.
    needs_pupil_fix = compute_pupil_stats and any(
        func is add_zscored_pupil_columns for func, _ in base_funcs
    )
    needs_seq_fix = any(
        func is create_fixation_sequence_tags for func, _ in group_funcs
    )

    fixations_df = None
    if needs_seq_fix or (needs_pupil_fix and pupil_fixations_path == fixations_path):
        if verbose:
            print(f"\nLoading fixations report once from: {fixations_path}")
        fixations_df = pd.read_csv(fixations_path)

    # Route the loaded fixations into the fixation-sequence group feature (it reads
    # raw fixation rows). Mirrors the pupil-stats injection for base funcs.
    if needs_seq_fix:
        group_funcs = [
            (func, {**kwargs, "fix_path": fixations_df})
            if func is create_fixation_sequence_tags
            else (func, kwargs)
            for func, kwargs in group_funcs
        ]

    # Resolve participant pupil stats once and inject them into the pupil
    # z-scoring base feature (only if that feature is actually being run). Reuse the
    # shared fixations_df when the pupil source is the canonical report; otherwise
    # get_participant_pupil_stats reads the explicit pupil_fixations_path override.
    if any(func is add_zscored_pupil_columns for func, _ in base_funcs):
        pupil_stats = get_participant_pupil_stats(
            stats_csv_path=pupil_stats_path,
            fixations_path=pupil_fixations_path,
            fixations=fixations_df if pupil_fixations_path == fixations_path else None,
            compute=compute_pupil_stats,
            verbose=verbose,
        )
        # Persist freshly-computed stats as an auxiliary artifact (no need to
        # re-save when they were just loaded from pupil_stats_path).
        if save_auxiliary and compute_pupil_stats:
            _save(pupil_stats, pupil_stats_path, label="participant pupils", verbose=verbose)
        base_funcs = [
            (func, {**kwargs, "pupil_stats": pupil_stats})
            if func is add_zscored_pupil_columns
            else (func, kwargs)
            for func, kwargs in base_funcs
        ]

    processed = _process(
        df_answers,
        base_funcs=base_funcs,
        group_funcs=group_funcs,
        add_last=add_last,
        add_rts=add_rts,
        last_labels_path=last_labels_path if save_auxiliary else None,
        rt_and_tfd_path=rt_and_tfd_path if save_auxiliary else None,
        button_clicks_path=button_clicks_path,
        include_paragraph=include_paragraph,
        label="all_participants",
        verbose=verbose,
    )

    _save(processed, output_path, label="all_participants", verbose=verbose)

    if split_column is not None:
        if verbose:
            print(f"\nSaving splits by column: {split_column}…")
        _save_splits(
            processed,
            split_column=split_column,
            base_output_path=output_path,
            split_output_paths=split_output_paths,
            verbose=verbose,
        )

    if verbose:
        print("\n✓ Done.\n")


if __name__ == "__main__":
    # Full rebuild from raw: writes all_participants.csv plus hunters.csv and
    # gatherers.csv (via the question-preview split) into L1_based_data, and the
    # auxiliary artifacts into L1_based_data/Auxiliary.
    main(
        split_column=C.QUESTION_PREVIEW_COLUMN,
        split_output_paths={
            True: HUNTERS_PROCESSED_PATH,
            False: GATHERERS_PROCESSED_PATH,
        },
    )


