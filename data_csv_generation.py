import pandas as pd
import numpy as np
import ast
from itertools import combinations
import itertools

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import mannwhitneyu

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

import networkx as nx
from matplotlib.lines import Line2D
import math


# ---------------------------------------------------------------------------
# Pre-existing column name constants
# ---------------------------------------------------------------------------

ARTICLE_COLUMN = 'article_id'
DIFFICULTY_COLUMN = 'difficulty_level'
BATCH_COLUMN = 'article_batch'
PARAGRAPH_COLUMN = 'paragraph_id'

REPEATED_TRIAL_COLUMN = "repeated_reading_trial"
PRACTICE_TRIAL_COLUMN = "practice_trial"
QUESTION_PREVIEW_COLUMN = "question_preview"

SELECTED_ANSWER_POSITION_COLUMN = "selected_answer_position"
CORRECT_ANSWER_POSITION_COLUMN = "correct_answer_position"
ANSWERS_ORDER_COLUMN = "answers_order"

INTEREST_AREA_ID = "IA_ID"
TRIAL_ID = 'TRIAL_INDEX'
PARTICIPANT_ID = 'participant_id'

# ---------------------------------------------------------------------------
# Created column name constants
# ---------------------------------------------------------------------------

TEXT_ID_COLUMN = "text_id"
IS_CORRECT_COLUMN = "is_correct"

AREA_SCREEN_LOCATION = "area_screen_loc"
AREA_LABEL_COLUMN = "area_label"
SELECTED_ANSWER_LABEL_COLUMN = "selected_answer_label"

# ---------------------------------------------------------------------------
# Helper Constants
# ---------------------------------------------------------------------------

ANSWER_PREFIX = "answer_"
ANSWER_LABELS = ["A", "B", "C", "D"]

#AREA_LABEL_CHOICES = ['question', 'answer_0', 'answer_1', 'answer_2', 'answer_3']
AREA_LABEL_CHOICES = ['question', 'top', 'left', 'right', 'bottom']

#DO NOT CHANGE
ANSWER_LABEL_CHOICES = ['question', 'answer_A', 'answer_B', 'answer_C', 'answer_D']


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
        df_filtered = df_filtered[df_filtered[REPEATED_TRIAL_COLUMN] == False].copy()
    if remove_practice:
        df_filtered = df_filtered[df_filtered[PRACTICE_TRIAL_COLUMN] == False].copy()

    df_hunters = df_filtered[df_filtered[QUESTION_PREVIEW_COLUMN] == True].copy()
    df_gatherers = df_filtered[df_filtered[QUESTION_PREVIEW_COLUMN] == False].copy()

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
    out[TEXT_ID_COLUMN] = (
            out[ARTICLE_COLUMN].astype(str) + '_' +
            out[DIFFICULTY_COLUMN].astype(str) + '_' +
            out[BATCH_COLUMN].astype(str) + '_' +
            out[PARAGRAPH_COLUMN].astype(str)
    )
    return out


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
    out[IS_CORRECT_COLUMN] = (
            out[SELECTED_ANSWER_POSITION_COLUMN]
            == out[CORRECT_ANSWER_POSITION_COLUMN]
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
        order = ast.literal_eval(row[ANSWERS_ORDER_COLUMN])
        answer_idx = order.index(label)
        return row[f"{ANSWER_PREFIX}{answer_idx + 1}"]

    for label in ANSWER_LABELS:
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

        index_id = group[INTEREST_AREA_ID] - 1

        conditions = [
            (index_id <= q_end),
            (index_id > q_end) & (index_id <= first_end),
            (index_id > first_end) & (index_id <= second_end),
            (index_id > second_end) & (index_id <= third_end),
            (index_id > third_end) & (index_id <= fourth_end)
        ]

        choices = AREA_LABEL_CHOICES
        group[AREA_SCREEN_LOCATION] = np.select(conditions, choices, default='unknown')
        return group

    df_area_split = df.set_index([TRIAL_ID, PARTICIPANT_ID]).groupby([TRIAL_ID, PARTICIPANT_ID], group_keys=False).apply(assign_area)
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
        loc = row[AREA_SCREEN_LOCATION]

        if loc == AREA_LABEL_CHOICES[0]:
            return "question"

        if loc in AREA_LABEL_CHOICES[1:]:
            try:
                # position index: 0..3 for answers
                pos_index = AREA_LABEL_CHOICES.index(loc) - 1
                answers_order = ast.literal_eval(row[ANSWERS_ORDER_COLUMN])
                letter = answers_order[pos_index]
                return letter_to_label.get(letter, None)
            except Exception:
                return None

        return None

    df_out[AREA_LABEL_COLUMN] = df_out.apply(get_area_label, axis=1)
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
    df[ANSWERS_ORDER_COLUMN] = df[ANSWERS_ORDER_COLUMN].apply(ast.literal_eval)
    df[SELECTED_ANSWER_LABEL_COLUMN] = df.apply(lambda row: row[ANSWERS_ORDER_COLUMN][row[SELECTED_ANSWER_POSITION_COLUMN]], axis=1)
    return df


# ---------------------------------------------------------------------------
#  Processing Pipelines
# ---------------------------------------------------------------------------


def add_base_features(df, functions, reset_index=True, verbose=False):
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
    if reset_index:
        out = out.reset_index(drop=True)
    return out



def main(
    ia_answers_path: str = "full/ia_A.csv",
    hunters_output_path: str = "output_data/base_features_hunters.csv",
    gatherers_output_path: str = "output_data/base_features_gatherers.csv",
    verbose: bool = True,
):
    """
    Full preprocessing pipeline:

    1. Load raw answers data.
    2. Split into hunters and gatherers (optionally removing repeats/practice).
    3. Add base feature set via a sequence of transformation functions.
    4. Save processed hunters and gatherers DataFrames as CSV.

    Parameters
    ----------
    ia_answers_path : str, optional
        Path to the raw answers CSV file.
    hunters_output_path : str, optional
        Output path for the processed hunters CSV.
    gatherers_output_path : str, optional
        Output path for the processed gatherers CSV.
    verbose : bool, optional
        If True, print progress information.
    """
    if verbose:
        print(f"Loading raw answers from: {ia_answers_path}")

    df_answers = load_raw_answers_data(ia_answers_path)

    if verbose:
        print("Splitting into hunters and gatherers...")

    df_hunters, df_gatherers = split_hunters_and_gatherers(df_answers)

    processing_functions = [
        add_text_id,
        add_is_correct,
        add_answer_text_columns,
        add_IA_screen_location,
        add_IA_answer_label,
        add_selected_answer_label,
    ]

    if verbose:
        print("Processing hunters DataFrame...")
    df_base_features_h = add_base_features(
        df_hunters, processing_functions, verbose=verbose
    )

    if verbose:
        print("Processing gatherers DataFrame...")
    df_base_features_g = add_base_features(
        df_gatherers, processing_functions, verbose=verbose
    )

    if verbose:
        print(f"Saving hunters features to: {hunters_output_path}")
    df_base_features_h.to_csv(hunters_output_path, index=False)

    if verbose:
        print(f"Saving gatherers features to: {gatherers_output_path}")
    df_base_features_g.to_csv(gatherers_output_path, index=False)

    if verbose:
        print("Done.")


if __name__ == "__main__":
    main()