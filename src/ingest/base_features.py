"""Row-level builders: the columns that make a raw report row a tidy row.

Each of these takes the IA frame and adds columns **in place on the row** --
nothing here aggregates. They parse the stimulus (text ids, the four answer
texts), resolve identity and the target (`is_correct`, the selected label), and
assign each interest area to a screen area.

**Why these live in `ingest/` and not `features/`.** They are not measurements.
`area_screen_loc`, `n_interest_areas` and the five `*_len` columns are the
*structure* of the tidy table, derived from the stimulus and the screen geometry
rather than from behaviour -- so by the boundary stated in both package
docstrings, this is ingest work. Before 2026-10-06 they sat in
`features/build.py`, which forced `features/` to import `ingest/geometry` and was
the one exception to the one-way layering rule (`restructure-map.md` S5.4).

> **One of them is arguable, and is flagged rather than quietly settled:**
> `add_total_answering_RT_normalized` divides a reading time by a word count,
> which is a *measurement*. It is here because it is row-level and reads
> `n_interest_areas` from `add_IA_screen_location` directly above it. Whether it
> moves to `features/` is a stage D question -- see the stage D proposal, "what I
> am least sure about".

Registered, and run in order, by `ingest/registry.py`.
"""

import ast

import numpy as np
import pandas as pd

from src.config import columns as C
from src.ingest.geometry import _reconcile_area_with_geometry


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

    A trial can never lack a confirmed selection -- the task does not let a
    participant leave one without confirming -- so a null position is a data
    fault, not a participant who abstained. Asserted rather than handled
    (`todo.md` T3.17b) because the equality below makes NaN compare unequal,
    which would score the fault as a *wrong answer* and put it in the model as
    a real incorrect trial. Verified 2026-09-27: zero nulls across all four
    datasets.
    """
    out = df.copy()

    for col in (C.SELECTED_ANSWER_POSITION_COLUMN, C.CORRECT_ANSWER_POSITION_COLUMN):
        null = out[col].isna()
        if null.any():
            # Name the trials, not just the count -- but this runs inside a base
            # feature, so fall back to whatever id columns exist rather than
            # turning an informative failure into a KeyError.
            id_cols = [c for c in (C.PARTICIPANT_ID, C.TRIAL_ID) if c in out.columns]
            offenders = out.loc[null, id_cols].drop_duplicates()
            raise ValueError(
                f"{col} is null on {int(null.sum())} interest-area row(s) "
                f"covering {len(offenders)} trial(s); is_correct would score "
                f"each as incorrect. First 5: "
                f"{offenders.head(5).to_dict('records')}"
            )

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
    - assigns each IA to one of LOC_CHOICES
    (ordered: question, answer on top, answer to the left, answer to the right, answer on bottom)

    That token-count assignment is then cross-checked against each interest
    area's on-screen rectangle, and the rectangle wins where they disagree --
    see `_reconcile_area_with_geometry`. The token counts are still computed and
    kept: the `*_len` columns are what `add_total_answering_RT_normalized` reads,
    and the comparison is what makes a stored text that does not match the
    display visible instead of silent.
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

    # The words that actually got an interest area. Every per-area metric is
    # computed over these, and add_total_answering_RT_normalized divides by them
    # too -- it is not always the stored token count above, because the stored
    # text does not always match what the screen rendered. Kept as a column so
    # the two counts stay comparable in the saved output.
    df["n_interest_areas"] = df.groupby([C.TRIAL_ID, C.PARTICIPANT_ID])[
        C.INTEREST_AREA_ID
    ].transform("size")

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
    return _reconcile_area_with_geometry(df_area_split)


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


def add_total_answering_RT_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total_answering_RT_normalized by dividing total_answering_RT by the
    number of words on the answer screen.

    The divisor is `n_interest_areas` -- the words that actually got an interest
    area -- rather than the stored-text token count in `total_words_on_screen`.
    The two agree on all but a handful of trials; where they differ the stored
    text carries a token the display never rendered, so normalising by it would
    divide the reading time by a word the tracker never measured. Both counts are
    kept in the output so the difference stays inspectable.
    """
    out = df.copy()

    len_cols = ["question_len", "1_len", "2_len", "3_len", "4_len"]

    out["total_words_on_screen"] = out[len_cols].sum(axis=1)

    out[C.TOTAL_ANSWERING_RT_NORMALIZED] = pd.to_numeric(
        out[C.CONFIRM_FINAL_ANSWER_RT], errors="coerce"
    ) / out["n_interest_areas"].replace(0, np.nan)

    n_differing = int((out["total_words_on_screen"] != out["n_interest_areas"]).sum())
    if n_differing:
        print(
            f"  answering RT: normalized by measured interest-area counts, which "
            f"differ from the stored token counts on {n_differing} interest-area row(s)"
        )

    return out

