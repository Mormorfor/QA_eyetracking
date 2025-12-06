import pandas as pd
import numpy as np
import ast
import os
import itertools
from src import constants as C


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

def load_raw_answers_data(ia_a_path = "full/ia_A.csv"):
    """
    Load raw interest area level answers data from CSV file.
    """
    return pd.read_csv(ia_a_path)


def load_raw_paragraphs_data(ia_p_path = "full/ia_P.csv"):
    """
    Load raw interest area level paragraphs data from CSV file.
    """
    return pd.read_csv(ia_p_path)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def split_hunters_and_gatherers(df, remove_repeats = True, remove_practice = True):
    """
    Split trials into 'hunters' and 'gatherers' based on question preview.
    Optionally removes repeated and practice trials before splitting.

    Parameters
    ----------
    df : DataFrame
        Interest-area level data.
    remove_repeats : bool, optional
        If True, remove repeated trials (REPEATED_TRIAL_COLUMN == True).
    remove_practice : bool, optional
        If True, remove practice trials (PRACTICE_TRIAL_COLUMN == True).

    Returns
    -------
    (DataFrame, DataFrame)
        Two DataFrames: (hunters, gatherers).
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


def add_text_id(df):
    """
    Add a unique text identifier column combining article, difficulty, batch, and paragraph.

    The new column is named according to TEXT_ID_COLUMN.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame. Must contain ARTICLE_COLUMN, DIFFICULTY_COLUMN,
        BATCH_COLUMN, and PARAGRAPH_COLUMN.

    Returns
    -------
    DataFrame
        Copy of the input DataFrame with an additional TEXT_ID_COLUMN.
    """
    out = df.copy()
    out[C.TEXT_ID_COLUMN] = (
            out[C.ARTICLE_COLUMN].astype(str) + '_' +
            out[C.DIFFICULTY_COLUMN].astype(str) + '_' +
            out[C.BATCH_COLUMN].astype(str) + '_' +
            out[C.PARAGRAPH_COLUMN].astype(str)
    )
    return out


def add_text_id_with_q(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'text_id_with_q' column that matches the original answer-text logic:

        text_id_with_q = text_id + '_' + same_critical_span

    This mirrors the old:
        article_id_difficulty_batch_paragraph_sameCriticalSpan
    """
    df = df.copy()

    df[C.TEXT_ID_WITH_Q_COLUMN] = (
        df[C.TEXT_ID_COLUMN].astype(str)
        + "_"
        + df[C.SAME_CRITICAL_SPAN_COLUMN].astype(str)
    )

    return df


def add_is_correct(df):
    """
    Adds IS_CORRECT_COLUMN to the dataframe based on comparison of selected and correct answer positions.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame. Must contain SELECTED_ANSWER_POSITION_COLUMN and
        CORRECT_ANSWER_POSITION_COLUMN.

    Returns
    -------
    DataFrame
        Copy of the input DataFrame with an additional IS_CORRECT_COLUMN
        (1 if selected == correct, 0 otherwise).
    """
    out = df.copy()
    out[C.IS_CORRECT_COLUMN] = (
            out[C.SELECTED_ANSWER_POSITION_COLUMN]
            == out[C.CORRECT_ANSWER_POSITION_COLUMN]
    ).astype(int)
    return out


def add_answer_text_columns(df):
    """
    Creates explicit answer text columns (answer_A, answer_B, answer_C, answer_D)
    per answer label (correctness level) and not just location.

    based on the answer order and the screen location based
    answer_1, answer_2, answer_3, answer_4 columns.

    This assumes that ANSWERS_ORDER_COLUMN contains a serialized list of labels
    like ['A', 'B', 'C', 'D'], and that the DataFrame has columns
    'answer_1', 'answer_2', 'answer_3', 'answer_4'.

    Parameters
    ----------
    df : DataFrame
        Interest-Area level data with ANSWERS_ORDER_COLUMN and answer_* columns.

    Returns
    -------
    DataFrame
        Copy of the input DataFrame with additional answer_X columns where X
        in ANSWER_LABELS.
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


def add_IA_screen_location(df):
    """
    Assign a screen-location label to each interest area within a trial.

    For each trial (TRIAL_ID, PARTICIPANT_ID), the function:
    - tokenizes question and answer_1–answer_4 text,
    - computes token lengths,
    - treats INTEREST_AREA_ID (1-based) as the token index,
    - assigns each IA to one of AREA_LABEL_CHOICES
    (ordered: question, answer on top, answer to the left, answer to the right, answer on bottom)

    Parameters
    ----------
    df : DataFrame
        Interest-area level data. Must contain:
        'question', 'answer_1'...'answer_4', INTEREST_AREA_ID, TRIAL_ID, PARTICIPANT_ID.

    Returns
    -------
    DataFrame
        Copy of the input DataFrame with an additional AREA_SCREEN_LOCATION column.
    """
    df = df.copy()
    for col in ['question', 'answer_1', 'answer_2', 'answer_3', 'answer_4']:
        df[col] = df[col].fillna('').astype(str)

    df['question_tokens'] = df['question'].str.split()
    df['1_tokens'] = df['answer_1'].str.split()
    df['2_tokens'] = df['answer_2'].str.split()
    df['3_tokens'] = df['answer_3'].str.split()
    df['4_tokens'] = df['answer_4'].str.split()

    df['question_len'] = df['question_tokens'].apply(len)
    df['1_len'] = df['1_tokens'].apply(len)
    df['2_len'] = df['2_tokens'].apply(len)
    df['3_len'] = df['3_tokens'].apply(len)
    df['4_len'] = df['4_tokens'].apply(len)

    def assign_area(group):
        q_len = group['question_len'].iloc[0]
        first_len = group['1_len'].iloc[0]
        second_len = group['2_len'].iloc[0]
        third_len = group['3_len'].iloc[0]
        fourth_len = group['4_len'].iloc[0]

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
            (index_id > third_end) & (index_id <= fourth_end)
        ]

        choices = C.AREA_LABEL_CHOICES
        group[C.AREA_SCREEN_LOCATION] = np.select(conditions, choices, default='unknown')
        return group

    df_area_split = df.set_index([C.TRIAL_ID, C.PARTICIPANT_ID]).groupby([C.TRIAL_ID, C.PARTICIPANT_ID], group_keys=False).apply(assign_area)
    return df_area_split


def add_IA_answer_label(df):
    """
    Add a logical answer label (correctness level) per interest area based on its screen location and
    the trial-specific answers order.

    Uses:
    - AREA_SCREEN_LOCATION: one of AREA_LABEL_CHOICES
    - ANSWERS_ORDER_COLUMN: serialized list like ['B', 'A', 'C', 'D']
     where index 0 corresponds to AREA_LABEL_CHOICES[1], index 1 to [2], etc.

    Logic
    -----
    - If AREA_SCREEN_LOCATION == AREA_LABEL_CHOICES[0], return 'question'.
    - Else, find position index p = AREA_LABEL_CHOICES.index(loc) - 1 (0..3),
     take letter = answers_order[p] (A/B/C/D),
     and map it to 'answer_A' / 'answer_B' / 'answer_C' / 'answer_D'.

    Parameters
    ----------
    df : DataFrame
       Must contain AREA_SCREEN_LOCATION and ANSWERS_ORDER_COLUMN.

    Returns
    -------
    DataFrame
       Copy of the input DataFrame with an additional AREA_LABEL_COLUMN.
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

        if loc == C.AREA_LABEL_CHOICES[0]:
            return "question"

        if loc in C.AREA_LABEL_CHOICES[1:]:
            try:
                # position index: 0..3 for answers
                pos_index = C.AREA_LABEL_CHOICES.index(loc) - 1
                answers_order = ast.literal_eval(row[C.ANSWERS_ORDER_COLUMN])
                letter = answers_order[pos_index]
                return letter_to_label.get(letter, None)
            except Exception:
                return None

        return None

    df_out[C.AREA_LABEL_COLUMN] = df_out.apply(get_area_label, axis=1)
    return df_out


def add_selected_answer_label(df):
    """
    Add a selected-answer-label (A/B/C/D) column based on the answer position
    and the trial-specific answer order.

    This converts a *location-based* selected answer (e.g. "selected position = 1")
    into a *label-based* answer (e.g. "selected answer = 'B'") by using the
    answers_order sequence stored per trial.

    Logic
    -----
    - ANSWERS_ORDER_COLUMN contains a serialized list like ['B', 'A', 'C', 'D'].
      These represent the labels shown on screen in order of:
          position 0 -> answers_order[0]
          position 1 -> answers_order[1]
          ...
    - SELECTED_ANSWER_POSITION_COLUMN is an integer 0–3 indicating which option
      the participant selected.
    - The function maps:
          selected_label = answers_order[selected_position]
      and stores it in SELECTED_ANSWER_LABEL_COLUMN.

    Parameters
    ----------
    df : DataFrame
        Must contain:
        - ANSWERS_ORDER_COLUMN (string representation of a list)
        - SELECTED_ANSWER_POSITION_COLUMN (integer index 0–3)

    Returns
    -------
    DataFrame
        Copy of the input DataFrame with an additional
        SELECTED_ANSWER_LABEL_COLUMN containing 'A', 'B', 'C', or 'D'.
    """
    df = df.copy()
    df[C.ANSWERS_ORDER_COLUMN] = df[C.ANSWERS_ORDER_COLUMN].apply(ast.literal_eval)
    df[C.SELECTED_ANSWER_LABEL_COLUMN] = (
        df.apply(lambda row: row[C.ANSWERS_ORDER_COLUMN][row[C.SELECTED_ANSWER_POSITION_COLUMN]], axis=1))
    return df

# ---------------------------------------------------------------------------
#  Group Features Creation
# ---------------------------------------------------------------------------

def create_mean_area_dwell_time(df):
    """
   Compute the mean dwell time per (trial, participant, area_label) group.

   This function aggregates interest-area–level data into area-level summaries
   by computing the mean dwell time for each unique combination of:
   - TRIAL_ID
   - PARTICIPANT_ID
   - AREA_LABEL_COLUMN (e.g., 'question', 'answer_A', ...)

   Logic
   -----
   The DataFrame is grouped by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN),
   and the mean of IA_DWELL_TIME is computed within each group. The resulting
   DataFrame contains one row per area per trial per participant.

   Parameters
   ----------
   df : DataFrame
       Must contain the columns:
       - TRIAL_ID
       - PARTICIPANT_ID
       - AREA_LABEL_COLUMN
       - IA_DWELL_TIME (numeric duration per interest area)

   Returns
   -------
   DataFrame
       A DataFrame with the following columns:
       - TRIAL_ID
       - PARTICIPANT_ID
       - AREA_LABEL_COLUMN
       - mean_dwell_time  (float)
   """
    return (
        df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False)
        .agg(**{C.MEAN_DWELL_TIME: (C.IA_DWELL_TIME, "mean")})
    )


def create_mean_area_fix_count(df):
    """
    Compute the mean number of fixations per (trial, participant, area_label) group.

    This function aggregates interest-area–level data by computing the mean
    number of fixations for each unique combination of:
    - TRIAL_ID
    - PARTICIPANT_ID
    - AREA_LABEL_COLUMN (e.g., 'question', 'answer_A', ...)

    Logic
    -----
    The DataFrame is grouped by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN),
    and the mean of IA_FIXATIONS_COUNT is computed within each group. The result
    is one row per area per trial per participant.

    Parameters
    ----------
    df : DataFrame
        Must contain the following columns:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - IA_FIXATIONS_COUNT (numeric)

    Returns
    -------
    DataFrame
        A DataFrame with columns:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - mean_fixations_count (float)
    """
    return (
        df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False)
        .agg(**{C.MEAN_FIXATIONS_COUNT: (C.IA_FIXATIONS_COUNT, "mean")})
    )

def create_mean_first_fix_duration(df):
    """
    Compute the mean first-fixation duration per (trial, participant, area_label).

    This function:
    Groups by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN).
    Computes the mean first-fixation duration within each group.

    Parameters
    ----------
    df : DataFrame
        Must contain:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - IA_FIRST_FIXATION_DURATION (may contain '.' for missing/zero values)

    Returns
    -------
    DataFrame
        A DataFrame containing:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - mean_first_fixation_duration (float)

        One row per area per trial per participant.
    """
    df[C.IA_FIRST_FIXATION_DURATION] = df[C.IA_FIRST_FIXATION_DURATION].replace('.', 0).astype(int)
    return (
        df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False)
        .agg(**{C.MEAN_FIRST_FIXATION_DURATION: (C.IA_FIRST_FIXATION_DURATION, "mean")})
    )


def create_skip_rate(df):
    """
    Compute the skip rate per (trial, participant, area_label).

    A skip is defined as an interest area (IA) with *zero dwell time*.
    The skip rate is the proportion of IAs within an area (e.g., 'answer_A')
    that were skipped by the participant during the trial.

    Logic
    -----
    - Create an indicator AREA_SKIPPED:
          1 if IA_DWELL_TIME == 0
          0 otherwise
    - Group by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN)
    - Compute the mean of AREA_SKIPPED → skip_rate

    Parameters
    ----------
    df : DataFrame
        Must contain:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - IA_DWELL_TIME (numeric)

    Returns
    -------
    DataFrame
        A DataFrame with:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - skip_rate (float in [0, 1])
    """
    df[C.AREA_SKIPPED] = (df[C.IA_DWELL_TIME] == 0).astype(int)
    return (
        df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False)
        .agg(**{C.SKIP_RATE: (C.AREA_SKIPPED, "mean")})
    )


def create_dwell_proportions(df):
    """
    Compute dwell time proportions per area within each trial and participant.

    For each (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN), this function:
    1. Sums IA_DWELL_TIME to obtain TOTAL_IA_DWELL_TIME per area.
    2. Sums TOTAL_IA_DWELL_TIME over all areas within a trial/participant to
       obtain TOTAL_TRIAL_DWELL_TIME.
    3. Computes AREA_DWELL_PROPORTION as:
           TOTAL_IA_DWELL_TIME / TOTAL_TRIAL_DWELL_TIME

    Any resulting NaN values (e.g., if TOTAL_TRIAL_DWELL_TIME is 0) are replaced by 0.

    Parameters
    ----------
    df : DataFrame
        Must contain:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - IA_DWELL_TIME (numeric)

    Returns
    -------
    DataFrame
        A DataFrame with columns:
        - TRIAL_ID
        - PARTICIPANT_ID
        - AREA_LABEL_COLUMN
        - TOTAL_IA_DWELL_TIME
        - TOTAL_TRIAL_DWELL_TIME
        - AREA_DWELL_PROPORTION
    """
    aggregated_df = (
        df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False)
        .agg({C.IA_DWELL_TIME: 'sum'})
        .rename(columns={C.IA_DWELL_TIME: C.TOTAL_IA_DWELL_TIME})
    )
    aggregated_df[C.TOTAL_TRIAL_DWELL_TIME] = (
        aggregated_df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID])[C.TOTAL_IA_DWELL_TIME].transform('sum'))
    aggregated_df[C.AREA_DWELL_PROPORTION] = aggregated_df[C.TOTAL_IA_DWELL_TIME] / aggregated_df[C.TOTAL_TRIAL_DWELL_TIME]
    aggregated_df = aggregated_df.fillna(0)

    return aggregated_df



def create_last_area_and_location_visited(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the last-visited *answer* area label and screen location per trial,
    using the most recent fixation on one of the answer areas.

    Logic
    -----
    1. Clean IA_LAST_FIXATION_TIME:
       - Replace '.' placeholder with 0
       - Convert to numeric (int)

    2. Sort by:
       - TRIAL_ID ascending
       - PARTICIPANT_ID ascending
       - IA_LAST_FIXATION_TIME descending
       (so the most recent fixation is first per group)

    3. Restrict to answer areas only
       (exclude 'question' = ANSWER_LABEL_CHOICES[0])

    4. Take the FIRST ROW per (TRIAL_ID, PARTICIPANT_ID)
       This is the last fixation on an answer area.

    5. Extract:
       - last_visited_label     (AREA_LABEL_COLUMN)
       - last_visited_location  (AREA_SCREEN_LOCATION)

    Returns
    -------
    DataFrame with one row per trial and participant:
        TRIAL_ID
        PARTICIPANT_ID
        last_visited_label
        last_visited_location

    Note
    ----
    If a trial never fixates any answer area (only 'question'), that trial
    will not appear in the result
    """
    df_local = df.copy()

    # 1) Clean fixation times
    df_local[C.IA_LAST_FIXATION_TIME] = (
        df_local[C.IA_LAST_FIXATION_TIME]
        .replace(".", 0)
        .astype(int)
    )

    # 2) Sort by trial/participant + descending fixation time
    df_sorted = df_local.sort_values(
        by=[C.TRIAL_ID, C.PARTICIPANT_ID, C.IA_LAST_FIXATION_TIME],
        ascending=[True, True, False],
    )

    # 3) Keep only answer areas (exclude 'question')
    answer_labels = [lab for lab in C.ANSWER_LABEL_CHOICES if lab != "question"]
    df_answers_only = df_sorted[df_sorted[C.AREA_LABEL_COLUMN].isin(answer_labels)]

    # 4) Take the most recent answer-fixation per trial/participant
    last_fix = (
        df_answers_only
        .groupby([C.TRIAL_ID, C.PARTICIPANT_ID], as_index=False)
        .head(1)
    )

    # 5) Rename to last-visited columns
    result = last_fix[
        [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN, C.AREA_SCREEN_LOCATION]
    ].rename(
        columns={
            C.AREA_LABEL_COLUMN: C.LAST_VISITED_LABEL,
            C.AREA_SCREEN_LOCATION: C.LAST_VISITED_LOCATION,
        }
    )

    return result.reset_index(drop=True)




def create_fixation_sequence_tags(df):
    """
    Build fixation sequences per trial/participant in terms of area labels and locations.

    For each (TRIAL_ID, PARTICIPANT_ID), this function:
    - Reads a precomputed fixation sequence stored in INTEREST_AREA_FIXATION_SEQUENCE,
      which is assumed to be a serialized list of IA_IDs (e.g. "[1, 2, 3, 2, 4]").
    - Maps each IA_ID to:
        * AREA_LABEL_COLUMN   (e.g. 'question', 'answer_A', ...)
        * AREA_SCREEN_LOCATION (e.g. 'top', 'left', ...) (or how they are defined in ANSWER_LABEL_CHOICES)
    - Produces two sequences:
        * FIX_SEQUENCE_BY_LABEL
        * FIX_SEQUENCE_BY_LOCATION

    Only IA_IDs that are present in the current group are used. The first element
    is dropped (label_sequence[1:], location_sequence[1:]) to exclude the very
    first fixation from the sequence (usually spillover fixation from question reading)

    Parameters
    ----------
    df : DataFrame
        Must contain:
        - TRIAL_ID
        - PARTICIPANT_ID
        - INTEREST_AREA_ID
        - AREA_LABEL_COLUMN
        - AREA_SCREEN_LOCATION
        - INTEREST_AREA_FIXATION_SEQUENCE

    Returns
    -------
    DataFrame
        One row per (TRIAL_ID, PARTICIPANT_ID) with:
        - TRIAL_ID
        - PARTICIPANT_ID
        - FIX_SEQUENCE_BY_LABEL    (list of labels)
        - FIX_SEQUENCE_BY_LOCATION (list of locations)
    """
    result = []
    for (trial_index, participant_id), group in df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID]):
        group_ids = set(group[C.INTEREST_AREA_ID].unique())

        id_to_label = dict(zip(group[C.INTEREST_AREA_ID], group[C.AREA_LABEL_COLUMN]))
        id_to_location = dict(zip(group[C.INTEREST_AREA_ID], group[C.AREA_SCREEN_LOCATION]))

        sequence_str = group[C.INTEREST_AREA_FIXATION_SEQUENCE].iloc[0]
        sequence = ast.literal_eval(sequence_str)

        label_sequence = []
        location_sequence = []

        for ia_id in sequence:
            if ia_id in group_ids:
                label_sequence.append(id_to_label[ia_id])
                location_sequence.append(id_to_location[ia_id])
        result.append({
            C.TRIAL_ID: trial_index,
            C.PARTICIPANT_ID: participant_id,
            C.FIX_SEQUENCE_BY_LABEL: label_sequence[1:],
            C.FIX_SEQUENCE_BY_LOCATION: location_sequence[1:]
        })

    return pd.DataFrame(result)


def create_simplified_fixation_tags(df):
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

    Parameters
    ----------
    df : DataFrame
        Must contain:
        - TRIAL_ID
        - PARTICIPANT_ID
        - INTEREST_AREA_ID
        - AREA_LABEL_COLUMN
        - AREA_SCREEN_LOCATION
        - INTEREST_AREA_FIXATION_SEQUENCE

    Returns
    -------
    DataFrame
        One row per (TRIAL_ID, PARTICIPANT_ID) with:
        - TRIAL_ID
        - PARTICIPANT_ID
        - SIMPLIFIED_FIX_SEQ_BY_LABEL    (tuple of labels)
        - SIMPLIFIED_FIX_SEQ_BY_LOCATION (tuple of locations)
    """
    result = []
    for (trial_index, participant_id), group in df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID]):
        group_ids = set(group[C.INTEREST_AREA_ID].unique())

        id_to_label = dict(zip(group[C.INTEREST_AREA_ID], group[C.AREA_LABEL_COLUMN]))
        id_to_location = dict(zip(group[C.INTEREST_AREA_ID], group[C.AREA_SCREEN_LOCATION]))

        sequence_str = group[C.INTEREST_AREA_FIXATION_SEQUENCE].iloc[0]
        sequence = ast.literal_eval(sequence_str)

        valid_fixations = []
        for ia_id in sequence:
            if ia_id in group_ids:
                valid_fixations.append((
                    ia_id,
                    id_to_label[ia_id],
                    id_to_location[ia_id],
                ))

        simpl_labels    = []
        simpl_locations = []

        for label, run in itertools.groupby(valid_fixations, key=lambda item: item[1]):
            run_list = list(run)
            simpl_labels.append(label)
            simpl_locations.append(run_list[0][2])

        result.append({
            C.TRIAL_ID :       trial_index,
            C.PARTICIPANT_ID:    participant_id,
            C.SIMPLIFIED_FIX_SEQ_BY_LABEL:    tuple(simpl_labels[:]),
            C.SIMPLIFIED_FIX_SEQ_BY_LOCATION:      tuple(simpl_locations[:]),
        })
    return pd.DataFrame(result)


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

    # Group-level functions
    "create_mean_area_dwell_time": {
        "callable": create_mean_area_dwell_time,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]},
        "kind": "group",
    },
    "create_mean_area_fix_count": {
        "callable": create_mean_area_fix_count,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]},
        "kind": "group",
    },
    "create_mean_first_fix_duration": {
        "callable": create_mean_first_fix_duration,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]},
        "kind": "group",
    },
    "create_skip_rate": {
        "callable": create_skip_rate,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]},
        "kind": "group",
    },
    "create_dwell_proportions": {
        "callable": create_dwell_proportions,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]},
        "kind": "group",
    },
    "create_last_area_and_location_visited": {
        "callable": create_last_area_and_location_visited,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "group",
    },
    "create_fixation_sequence_tags": {
        "callable": create_fixation_sequence_tags,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "group",
    },
    "create_simplified_fixation_tags": {
        "callable": create_simplified_fixation_tags,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "group",
    },
}



def resolve_base_functions(name_list=None):
    """
    Resolve base feature functions.

    If name_list is None, return ALL base functions from FUNCTION_REGISTRY
    in registry insertion order. Otherwise, return only the named ones.
    """
    # Case 1: no explicit list → use all base functions
    if name_list is None:
        return [
            entry["callable"]
            for name, entry in FUNCTION_REGISTRY.items()
            if entry.get("kind") == "base"
        ]

    # Case 2: explicit list → validate and return only those
    resolved = []
    for name in name_list:
        if name not in FUNCTION_REGISTRY:
            raise ValueError(f"Unknown base function: {name}")
        entry = FUNCTION_REGISTRY[name]
        if entry.get("kind") != "base":
            raise ValueError(f"Function '{name}' is not registered as a base feature.")
        resolved.append(entry["callable"])
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
        # Simple string: just a function name
        if isinstance(item, str):
            name = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "group":
                raise ValueError(f"Function '{name}' is not registered as a group feature.")
            resolved.append((entry["callable"], entry.get("default_kwargs", {})))
            continue

        # Tuple: (function_name, user_kwargs)
        if isinstance(item, tuple) and len(item) == 2:
            name, user_kwargs = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "group":
                raise ValueError(f"Function '{name}' is not registered as a group feature.")

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

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    functions : list of callables
        List of functions to apply. Each must have signature:
            func(df: DataFrame) -> DataFrame
    reset_index : bool, optional
        If True (default), reset the index of the final DataFrame before
        returning it.
    verbose : bool, optional
        If True, print the name of each function as it is applied.

    Returns
    -------
    DataFrame
        The transformed DataFrame after all functions have been applied.
    """
    out = df.copy()
    for func in functions:
        if verbose:
            print(f"Running: {func.__name__}")
        out = func(out)
    return out.reset_index()


def generate_new_row_features(functions, df, default_join_columns=None,
                              verbose=True):
    """
    Iteratively compute and merge group-level features into a row-level DataFrame.

    Each entry in `functions` is a tuple:
        (func, func_kwargs)

    where:
        - func is a callable with signature: func(df: DataFrame) -> DataFrame
        - func_kwargs is a dict that may contain:
            * 'join_columns': list of columns to use for merging
              (defaults to [TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN])

    For each function:
    1. Compute `new_features_df = func(result_df)`
    2. Merge `new_features_df` into `result_df` using a left join on `join_columns`.

    This allows chaining multiple group-level feature generators that return
    aggregated DataFrames, and incrementally enrich the original row-level
    data with new columns.

    Parameters
    ----------
    functions : list of (callable, dict)
        List of (function, kwargs) tuples. Each function must accept a single
        DataFrame and return a DataFrame with grouping columns + new features.
    df : DataFrame
        Base DataFrame to enrich with new features.
    default_join_columns : list of str, optional
        Default columns to use for merging results of each function.
        If None, defaults to [TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN].
    verbose : bool, optional
        If True, print the name of each function as it is applied.

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

        join_columns = func_kwargs.get('join_columns', default_join_columns)

        new_features_df = func(result_df)
        result_df = result_df.merge(new_features_df, on=join_columns, how='left')

    return result_df

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    ia_answers_path: str = "data_raw/full/ia_A.csv",
    hunters_output_path: str = "data/hunters.csv",
    gatherers_output_path: str = "data/gatherers.csv",
    base_function_names: list = None,
    group_function_names: list = None,
    verbose: bool = True,
):
    """
    Full preprocessing pipeline:

    1. Load raw answers data.
    2. Split into hunters and gatherers.
    3. Apply base per-row features.
    4. Apply group-level features.
    5. Save output CSV files.

    Parameters
    ----------
    ia_answers_path : str
        Path to the raw answers CSV.
    hunters_output_path : str
        Output file path for hunter trials.
    gatherers_output_path : str
        Output file path for gatherer trials.
    base_function_names : list or None
        Optional override list of base-feature function names.
        If None, defaults from FUNCTION_REGISTRY are used.
    group_function_names : list or None
        Optional override list for group-level functions.
        If None, defaults from FUNCTION_REGISTRY are used.
    verbose : bool
        If True, prints progress.
    """

    os.makedirs(os.path.dirname(hunters_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(gatherers_output_path), exist_ok=True)

    if verbose:
        print(f"\nLoading raw answers from: {ia_answers_path}")

    df_answers = load_raw_answers_data(ia_answers_path)

    if verbose:
        print("Splitting into hunters and gatherers…")

    df_hunters, df_gatherers = split_hunters_and_gatherers(df_answers)

    if verbose:
        print("\nResolving processing function lists…")

    base_funcs = resolve_base_functions(base_function_names)
    group_funcs = resolve_group_functions(group_function_names)

    # Hunters
    if verbose:
        print("\nProcessing hunters (row-level)…")
    df_h = add_base_features(df_hunters, base_funcs, verbose=verbose)

    if verbose:
        print("Applying group-level features for hunters…")
    df_h = generate_new_row_features(group_funcs, df_h)

    if verbose:
        print(f"Saving hunters features to: {hunters_output_path}")
    df_h.to_csv(hunters_output_path, index=False)

    # Gatherers
    if verbose:
        print("\nProcessing gatherers (row-level)…")
    df_g = add_base_features(df_gatherers, base_funcs, verbose=verbose)

    if verbose:
        print("Applying group-level features for gatherers…")
    df_g = generate_new_row_features(group_funcs, df_g)

    if verbose:
        print(f"Saving gatherers features to: {gatherers_output_path}")
    df_g.to_csv(gatherers_output_path, index=False)

    if verbose:
        print("\n✓ Done.\n")



if __name__ == "__main__":
    main()
