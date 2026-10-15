"""Per-word columns that turn a raw report row into a tidy one.

Each function takes the word-level frame and adds columns to every row; none
of them summarises anything. Between them they:

- name the item: `text_id`, `text_id_with_q`
- score the answer: `is_correct`, `selected_answer_label`
- put each answer's text under its letter: `answer_A`..`answer_D`
- place each word in a screen area: `area_screen_loc`, then `area_label`
- scale the total answering time by the words on screen:
  `total_answering_RT_normalized`

They describe the trial and its text rather than the eye movements, which is
why they are in `ingest/` rather than `features/`. They run in the order
listed in `ingest/registry.py`, and some need columns made by earlier ones.
"""

import ast

import numpy as np
import pandas as pd

from src.config import columns as C
from src.ingest.geometry import _reconcile_area_with_geometry


def add_text_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add `text_id`, naming the paragraph: batch_article_difficulty_paragraph."""
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
    """Add `text_id_with_q` = text_id + "_" + same_critical_span.

    That is the paragraph plus which of its three questions was asked -- one
    test item.
    """
    df = df.copy()

    df[C.TEXT_ID_WITH_Q_COLUMN] = (
        df[C.TEXT_ID_COLUMN].astype(str)
        + "_"
        + df[C.SAME_CRITICAL_SPAN_COLUMN].astype(str)
    )

    return df


def add_is_correct(df: pd.DataFrame) -> pd.DataFrame:
    """Add `is_correct`: 1 if the chosen position is the correct one, else 0.

    Stops with an error if either position is missing. A participant cannot
    leave a trial without confirming an answer, so a missing one is a data
    fault -- and the comparison below would quietly score it as a wrong answer.
    """
    out = df.copy()

    for col in (C.SELECTED_ANSWER_POSITION_COLUMN, C.CORRECT_ANSWER_POSITION_COLUMN):
        null = out[col].isna()
        if null.any():
            # List the trials involved, using whichever id columns are present.
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
    """Put each answer's text under its letter: `answer_A`..`answer_D`.

    The report stores the text by screen position (`answer_1`..`answer_4`);
    `answers_order` says which letter was in each position.
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
    """Add `area_screen_loc`: which screen area each word is in (question, top,
    left, right or bottom answer).

    Done two ways, and the second wins:

    1. by counting words: split the stored question and answer texts into
       words, and use each word's number (`IA_ID`) to see which text it falls in;
    2. by position on screen, in `geometry._reconcile_area_with_geometry`, which
       replaces (1) wherever the two disagree and prints the corrections.

    The word counts are kept as columns (`question_len`, `1_len`..`4_len`, and
    the word lists), and so is `n_interest_areas`, the number of words that
    actually appeared on screen.
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

    # The number of words that actually appeared on screen in this trial. It can
    # differ from the word count of the stored text, because the stored text
    # does not always match what was displayed. Every per-area measurement is
    # over these words, and add_total_answering_RT_normalized divides by this.
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
    """Add `area_label`: each word's area by meaning (question, answer_A..D)
    rather than by place.

    The question stays "question". For an answer, its position number is its
    place in LOC_CHOICES minus 1 (0-3), and `answers_order` says which letter
    sat in that position on this trial.
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
    """Add `selected_answer_label`: the letter (A-D) the participant chose.

    Turns the chosen screen position (e.g. 1) into a letter (e.g. "B") using
    `answers_order`.

    Careful: this also turns `answers_order` itself from text into a list, in
    place. Features that read `answers_order` as text must run before this one.
    """
    df = df.copy()
    df[C.ANSWERS_ORDER_COLUMN] = df[C.ANSWERS_ORDER_COLUMN].apply(ast.literal_eval)
    df[C.SELECTED_ANSWER_LABEL_COLUMN] = df.apply(
        lambda row: row[C.ANSWERS_ORDER_COLUMN][row[C.SELECTED_ANSWER_POSITION_COLUMN]],
        axis=1,
    )
    return df


def add_total_answering_RT_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Add `total_answering_RT_normalized`: the time from the answers appearing
    to the confirming press, divided by the number of words on screen.

    It divides by `n_interest_areas` (the words actually displayed), not by
    `total_words_on_screen` (the word count of the stored text). The two
    differ on a handful of trials, where the stored text has a word the screen
    never showed. Both counts are kept, and the number of rows where they
    differ is printed.
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

