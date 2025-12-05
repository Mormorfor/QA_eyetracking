"""
sequence_visualisations.py

Utilities for visualising fixation sequences (by label or by screen location)
using the processed eyetracking data produced by data_csv_generation.py.

Typical usage in a notebook
---------------------------
import pandas as pd
import sequence_visualisations as SV

# 1) Load processed hunters / gatherers
df_h = pd.read_csv("output_data/hunters.csv")
df_g = pd.read_csv("output_data/gatherers.csv")

# 2) Build one-row-per-trial table with sequences
data_rows_h = SV.build_sequence_rows(
    df_h,
    representation="raw",         # or "simplified"
    include_question=True         # or False to drop 'question' tokens
)

# 3) Visualise random participant / text
SV.visualize_by_id(
    data_rows_h,
    by_person_or_text="person",   # 'person' or 'text'
    fix_by="loc",                 # 'label' or 'loc'
    identifier="l11_525"          # or None for random
)
"""

import ast
import random
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src import constants as C

# ============================================================================
# Category / legend definitions
# ============================================================================

# Colors for AREA_LABEL-style sequences (logical answers: question, answer_A, ...)
CATEGORIES_LABEL = {
    "question": "#74a9cf",
    "answer_A": "#238b45",
    "answer_B": "#74c476",
    "answer_C": "#bae4b3",
    "answer_D": "#edf8e9",

    # Colours for the *letter* labels (selected/correct) in the margin
    "A": "green",
    "B": "gold",
    "C": "darkorange",
    "D": "red",

    # Fallbacks / noise
    "out_of_bounds": "white",
    "unknown": "white",
    None: "white",
}

LEGEND_MAPPING_LABEL = {
    "question": "Question",
    "answer_A": "Answer A",
    "answer_B": "Answer B",
    "answer_C": "Answer C",
    "answer_D": "Answer D",
    "out_of_bounds": None,
    "unknown": None,

    # We *don't* want A/B/C/D letter entries in the legend
    "A": None,
    "B": None,
    "C": None,
    "D": None,
    None: None,
}


# Colors for AREA_SCREEN_LOCATION-style sequences
# (question, answer_0(top), answer_1(left), answer_2(right), answer_3(bottom))
# Use constants.AREA_LABEL_CHOICES to keep it in sync.
_LOC_CHOICES = C.AREA_LABEL_CHOICES  # ['question', 'answer_0(top)', ..., ...]

CATEGORIES_LOC = {
    "question": _LOC_CHOICES[0] and "#74a9cf",

    # Answers by on-screen position
    _LOC_CHOICES[1]: "#ffffb2",   # top answer on screen
    _LOC_CHOICES[2]: "#fecc5c",   # right
    _LOC_CHOICES[3]: "#fd8d3c",   # bottom
    _LOC_CHOICES[4]: "#e31a1c",   # left

    # Letter labels for margin
    "A": "green",
    "B": "gold",
    "C": "darkorange",
    "D": "red",

    "out_of_bounds": 'white',
    "unknown": 'white',
}

LEGEND_MAPPING_LOC = {
    "out_of_bounds": None,
    "unknown": None,
    "question": "Question",
    _LOC_CHOICES[1]: "Top answer on screen",
    _LOC_CHOICES[2]: "Right answer on screen",
    _LOC_CHOICES[3]: "Bottom answer on screen",
    _LOC_CHOICES[4]: "Left answer on screen",

    "A": None,
    "B": None,
    "C": None,
    "D": None,
    None: None,
}


# ============================================================================
# Helpers
# ============================================================================

def _parse_sequence(value) -> List:
    """
    Ensure 'value' is a Python list/tuple, not a string representation.

    Handles:
    - already-a-list/tuple
    - string like "['question', 'answer_A']"
    - NaN / None -> empty list
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            return [value]
    return [value]


def _filter_question_from_pair(
    labels: Iterable,
    locs: Iterable,
    question_token: str = "question"
) -> Tuple[List, List]:
    """
    Remove 'question' tokens from a (labels, locs) pair of sequences,
    preserving alignment.
    """
    new_labels = []
    new_locs = []
    for lab, loc in zip(labels, locs):
        if lab == question_token:
            continue
        new_labels.append(lab)
        new_locs.append(loc)
    return new_labels, new_locs


# ============================================================================
# Building the sequence table from processed data
# ============================================================================

def build_sequence_rows(
    df: pd.DataFrame,
    representation: str = "raw",
    include_question: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Build a compact table (one row per TRIAL/PARTICIPANT) with
    fixation sequences and key labels, ready for visualisation.

    Parameters
    ----------
    df : DataFrame
        Processed data (e.g. hunters.csv / gatherers.csv) that already
        includes the sequence columns from data_csv_generation:
            - C.FIX_SEQUENCE_BY_LABEL / C.FIX_SEQUENCE_BY_LOCATION
            - C.SIMPLIFIED_FIX_SEQ_BY_LABEL / C.SIMPLIFIED_FIX_SEQ_BY_LOCATION
    representation : {'raw', 'simplified'}
        Which sequence variant to use:
            'raw'        -> FIX_SEQUENCE_BY_*
            'simplified' -> SIMPLIFIED_FIX_SEQ_BY_*
    include_question : bool
        If False, all 'question' tokens are removed from the sequences
        on-the-fly (no additional columns are created).
    drop_duplicates : bool
        If True, drop duplicate rows (useful if the source df is still
        IA-level and not already aggregated).

    Returns
    -------
    DataFrame
        Columns:
            - C.TRIAL_ID
            - C.PARTICIPANT_ID
            - C.TEXT_ID_COLUMN
            - C.FIX_SEQUENCE_BY_LABEL  (tuple)
            - C.FIX_SEQUENCE_BY_LOCATION (tuple)
            - C.SELECTED_ANSWER_LABEL_COLUMN
            - C.CORRECT_ANSWER_POSITION_COLUMN
            - C.IS_CORRECT_COLUMN (if present in df)

        Note
        ----
        The *names* of the output sequence columns are always
        C.FIX_SEQUENCE_BY_LABEL / C.FIX_SEQUENCE_BY_LOCATION, regardless
        of whether 'raw' or 'simplified' representation is used.
        The 'representation' flag only controls which source columns
        are read from.
    """
    if representation not in {"raw", "simplified"}:
        raise ValueError("representation must be 'raw' or 'simplified'")

    # Which *source* columns do we read from?
    if representation == "raw":
        label_col_src = C.FIX_SEQUENCE_BY_LABEL
        loc_col_src   = C.FIX_SEQUENCE_BY_LOCATION
    else:
        label_col_src = C.SIMPLIFIED_FIX_SEQ_BY_LABEL
        loc_col_src   = C.SIMPLIFIED_FIX_SEQ_BY_LOCATION

    # Select the minimal set of columns we actually need
    cols = [
        C.TRIAL_ID,
        C.PARTICIPANT_ID,
        C.TEXT_ID_COLUMN,
        label_col_src,
        loc_col_src,
        C.SELECTED_ANSWER_LABEL_COLUMN,
        C.CORRECT_ANSWER_POSITION_COLUMN,
    ]

    if C.IS_CORRECT_COLUMN in df.columns:
        cols.append(C.IS_CORRECT_COLUMN)

    seq_df = df[cols].copy()

    if drop_duplicates:
        seq_df = seq_df.drop_duplicates()

    # Normalise sequences (parse from strings if necessary)
    seq_df[label_col_src] = seq_df[label_col_src].apply(_parse_sequence)
    seq_df[loc_col_src]   = seq_df[loc_col_src].apply(_parse_sequence)

    # Optionally remove 'question' tokens
    def _process_row(row):
        labels = row[label_col_src]
        locs   = row[loc_col_src]
        if not include_question:
            labels, locs = _filter_question_from_pair(labels, locs)
        return pd.Series(
            {
                C.FIX_SEQUENCE_BY_LABEL:    tuple(labels),
                C.FIX_SEQUENCE_BY_LOCATION: tuple(locs),
            }
        )

    seq_df[[C.FIX_SEQUENCE_BY_LABEL, C.FIX_SEQUENCE_BY_LOCATION]] = seq_df.apply(
        _process_row, axis=1
    )

    # Keep only the unified columns
    keep_cols = [
        C.TRIAL_ID,
        C.PARTICIPANT_ID,
        C.TEXT_ID_COLUMN,
        C.FIX_SEQUENCE_BY_LABEL,
        C.FIX_SEQUENCE_BY_LOCATION,
        C.SELECTED_ANSWER_LABEL_COLUMN,
        C.CORRECT_ANSWER_POSITION_COLUMN,
    ]
    if C.IS_CORRECT_COLUMN in seq_df.columns:
        keep_cols.append(C.IS_CORRECT_COLUMN)

    return seq_df[keep_cols].reset_index(drop=True)



# ============================================================================
# Core plotting helpers
# ============================================================================

def visualize_stacked_rows_with_two_labels(
    data_rows: List[Iterable],
    categories: dict,
    selected_answer_labels: List,
    additional_labels: List,
    num_rows: Optional[int] = 100,
    start_index: int = 0,
    legend_mapping: Optional[dict] = None,
):
    """
    Visualise multiple sequences as stacked horizontal bars.

    Each sequence row is coloured according to `categories`. For each row,
    we print two labels to the left:
        - selected_answer_labels[i]
        - additional_labels[i]    (e.g. correct_answer_position)

    Parameters
    ----------
    data_rows : list of sequences
        Each element is a list/tuple of tokens (area labels or locations).
    categories : dict
        Mapping from token -> colour string.
    selected_answer_labels : list
        One label per row (e.g. 'A', 'B', 'C', 'D').
    additional_labels : list
        Second label per row (e.g. correct-answer position).
    num_rows : int or None
        How many rows to plot (starting from start_index). If None, plot all.
    start_index : int
        Index in data_rows at which to start.
    legend_mapping : dict or None
        Mapping original_token -> legend label (or None to hide).
    """
    # Slice requested window
    if num_rows is None:
        selected_rows = data_rows[start_index:]
        selected_labels = selected_answer_labels[start_index:]
        additional_labels = additional_labels[start_index:]
    else:
        selected_rows = data_rows[start_index:start_index + num_rows]
        selected_labels = selected_answer_labels[start_index:start_index + num_rows]
        additional_labels = additional_labels[start_index:start_index + num_rows]

    if not selected_rows:
        print("No rows to visualise.")
        return

    max_length = max((len(row) for row in selected_rows), default=0) + 2

    color_data = []
    for row in selected_rows:
        row = list(row)
        color_row = [categories.get(value, "gray") for value in row]
        # pad to same length
        if len(color_row) < max_length:
            color_row += ["white"] * (max_length - len(color_row))
        color_data.append(color_row)

    plt.figure(figsize=(15, max(2, len(selected_rows) * 0.3)))

    for i, (color_row, label, additional_label) in enumerate(
        zip(color_data, selected_labels, additional_labels)
    ):
        plt.bar(
            range(max_length),
            [1] * max_length,
            color=color_row,
            width=1.0,
            edgecolor="none",
            bottom=i,
        )
        # additional_label (e.g. correct answer) on the far left
        plt.text(
            -3.5,
            i + 0.5,
            str(additional_label),
            va="center",
            ha="right",
            fontsize=10,
            color=categories.get(additional_label, "black"),
        )
        # selected answer label just left of the sequence
        plt.text(
            -1.5,
            i + 0.5,
            str(label),
            va="center",
            ha="right",
            fontsize=10,
            color=categories.get(label, "black"),
        )

    plt.axis("off")

    # Legend
    if legend_mapping:
        legend_handles = [
            mpatches.Patch(
                color=categories.get(original_label, "gray"),
                label=new_label,
            )
            for original_label, new_label in legend_mapping.items()
            if new_label is not None
        ]
    else:
        legend_handles = [
            mpatches.Patch(color=color, label=str(label))
            for label, color in categories.items()
        ]

    if legend_handles:
        plt.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Categories",
            fontsize=14,
            title_fontsize=16,
        )

    plt.tight_layout()
    plt.show()


# ============================================================================
# Public visualisation API
# ============================================================================

def visualize_by_id(
    data: pd.DataFrame,
    by_person_or_text: str,
    fix_by: str,
    identifier: Optional[str] = None,
    max_rows: Optional[int] = None,
):
    """
    High-level helper to visualise fixation sequences for a single participant
    or a single text.

    This assumes that `data` is the *sequence table* returned by
    `build_sequence_rows`, i.e. has columns:
        - C.PARTICIPANT_ID
        - C.TEXT_ID_COLUMN
        - 'fix_by_label'
        - 'fix_by_loc'
        - C.SELECTED_ANSWER_LABEL_COLUMN
        - C.CORRECT_ANSWER_POSITION_COLUMN

    Parameters
    ----------
    data : DataFrame
        Output of build_sequence_rows(...).
    by_person_or_text : {'person', 'text'}
        Whether to filter by participant_id or text_id.
    fix_by : {'label', 'loc'}
        Whether to colour by AREA_LABEL ('question', 'answer_A', ...)
        or by AREA_SCREEN_LOCATION (C.AREA_LABEL_CHOICES entries).
    identifier : str or None
        The participant_id or text_id to visualise. If None, one is
        chosen at random from the available values.
    max_rows : int or None
        Maximum number of trials/rows to display. If None, show all
        trials for that id.
    """
    if by_person_or_text not in {"person", "text"}:
        raise ValueError("by_person_or_text must be either 'person' or 'text'.")

    if fix_by not in {"label", "loc"}:
        raise ValueError("fix_by must be either 'label' or 'loc'.")

    if fix_by == "label":
        categories = CATEGORIES_LABEL
        legend_mapping = LEGEND_MAPPING_LABEL
        data_column = C.FIX_SEQUENCE_BY_LABEL
    else:
        categories = CATEGORIES_LOC
        legend_mapping = LEGEND_MAPPING_LOC
        data_column = C.FIX_SEQUENCE_BY_LOCATION

    filter_column = (
        C.PARTICIPANT_ID if by_person_or_text == "person" else C.TEXT_ID_COLUMN
    )

    if identifier is None:
        available_ids = data[filter_column].dropna().unique()
        if len(available_ids) == 0:
            raise ValueError(f"No valid entries found in column '{filter_column}'.")
        identifier = random.choice(available_ids)
        print(f"Randomly selected {filter_column}: {identifier}")

    subset = data[data[filter_column] == identifier]
    if subset.empty:
        print(f"No data found for {filter_column} = {identifier}.")
        return

    if data_column not in subset.columns:
        raise KeyError(f"Column '{data_column}' not found in the data.")

    fix_data_rows = subset[data_column].tolist()
    selected_answer_labels = subset[C.SELECTED_ANSWER_LABEL_COLUMN].tolist()
    additional_labels = subset[C.CORRECT_ANSWER_POSITION_COLUMN].tolist()

    visualize_stacked_rows_with_two_labels(
        fix_data_rows,
        categories,
        selected_answer_labels,
        additional_labels,
        num_rows=max_rows or len(fix_data_rows),
        legend_mapping=legend_mapping,
    )
