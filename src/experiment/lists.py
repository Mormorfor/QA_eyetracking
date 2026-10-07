"""Building the Study 2 counterbalanced lists.

A double Latin square -- regime x article group, then question ordering -- giving 27 lists,
times 6 regime orderings = **54 ordered lists**. Those CSVs are the hand-off to the
presentation software, which is not in this repo.

Lifted from `experiment_builder/lists_builder.ipynb` in stage F. The functions moved; the
orchestration that ran them stays in `notebooks/experiment/lists_builder.ipynb`.
"""

from pathlib import Path
import sys
import os
import pandas as pd
import src.config.columns as Con
from src.config.datasets import EXPERIMENT_TEXT_COMPLETED_PATH, DATA_DIR, EXPERIMENT_DIR
from ftfy import fix_text
from itertools import permutations
import random
import numpy as np


REGIMES = ["full knowledge", "partial knowledge", "no knowledge"]

ARTICLE_GROUPS = {
    "group_1": [1, 2, 3],
    "group_2": [4, 5, 6],
    "group_3": [7, 8, 9],
}

LATIN_SQUARE = [
    ["full knowledge", "partial knowledge", "no knowledge"],
    ["partial knowledge", "no knowledge", "full knowledge"],
    ["no knowledge", "full knowledge", "partial knowledge"],
]

LIST_NAMES = {
    1: "list_A",
    2: "list_B",
    3: "list_C",
}

QUESTION_LATIN_SQUARE = {
    "q0": [0, 1, 2],
    "q1": [1, 2, 0],
    "q2": [2, 0, 1],
}

REGIMES = ["full knowledge", "partial knowledge", "no knowledge"]

REGIME_ORDERINGS = {
    f"O{i}": list(order)
    for i, order in enumerate(permutations(REGIMES), start=1)
}

answer_cols = ["answer_A", "answer_B", "answer_C", "answer_D"]

rng = np.random.default_rng(seed=42) 


def repair_mojibake(value):
    if isinstance(value, str):
        return fix_text(value)
    return value


def clean_batch_articles(df):
    df = df.copy()

    # Normalize article_id to string for robust comparison
    article_id_str = df["article_id"].astype(str).str.strip()

    # Remove article_id 0 / article_0 / id_0
    df = df[~article_id_str.isin(["0", "article_0", "id_0"])].copy()

    # Recompute normalized article_id after filtering
    article_id_str = df["article_id"].astype(str).str.strip()

    # article_10 is the practice article
    df["practice"] = article_id_str.isin(["10", "article_10", "id_10"]).astype(int)

    return df


def restrict_practice_questions(df):
    df = df.copy()

    # Robust numeric versions, in case these columns were read as strings
    paragraph_num = pd.to_numeric(df["paragraph_id"], errors="coerce")
    question_num = pd.to_numeric(df["onestopqa_question_id"], errors="coerce")

    # Keep all non-practice rows
    non_practice_mask = df["practice"] == 0

    # For practice rows, keep only:
    # paragraphs 1, 2, 3
    # question 0
    practice_keep_mask = (
        (df["practice"] == 1)
        & (paragraph_num.isin([1, 2, 3]))
        & (question_num == 0)
    )

    return df[non_practice_mask | practice_keep_mask].copy()


def build_latin_square_lists_for_batch(df, batch_name):
    """
    Creates 3 experimental regime lists for one batch.

    Non-practice articles 1-9 are assigned to regimes by Latin square.
    Practice article 10 is duplicated into all regimes in every list.
    """

    df = df.copy()

    experimental_rows = df[df["practice"] == 0].copy()
    practice_rows = df[df["practice"] == 1].copy()

    all_list_parts = []

    for list_idx, regime_assignment in enumerate(LATIN_SQUARE, start=1):

        list_name = LIST_NAMES[list_idx]

        # 1. Add experimental articles according to Latin square
        for group_idx, (group_name, article_ids) in enumerate(ARTICLE_GROUPS.items()):

            regime = regime_assignment[group_idx]

            group_rows = experimental_rows[
                experimental_rows["article_id"].isin(article_ids)
            ].copy()

            group_rows["experiment_list"] = f"{batch_name}_{list_name}"
            group_rows["regime_list"] = list_name
            group_rows["regime"] = regime
            group_rows["article_group"] = group_name

            all_list_parts.append(group_rows)

        # 2. Duplicate practice rows into all three regimes
        for regime in REGIMES:
            practice_copy = practice_rows.copy()

            practice_copy["experiment_list"] = f"{batch_name}_{list_name}"
            practice_copy["regime_list"] = list_name
            practice_copy["regime"] = regime
            practice_copy["article_group"] = "practice"

            all_list_parts.append(practice_copy)

    batch_lists = pd.concat(all_list_parts, ignore_index=True)

    return batch_lists


def add_question_lists_to_regime_lists(df):
    """
    Takes already regime-separated experiment lists and creates
    three question-list versions of each regime list.

    For non-practice rows:
    - keeps one question per article-paragraph
    - rotates question choice by paragraph position

    For practice rows:
    - keeps practice rows as they are
    - duplicates them into each question list
    """

    all_parts = []

    for question_list_name, question_assignment in QUESTION_LATIN_SQUARE.items():

        question_version = df.copy()

        non_practice = question_version[question_version["practice"] == 0].copy()
        practice = question_version[question_version["practice"] == 1].copy()

        # paragraph_id 1,2,3,... becomes repeating positions 0,1,2,0,1,2...
        paragraph_position = (non_practice["paragraph_id"] - 1) % 3

        selected_question = paragraph_position.map(
            {
                0: question_assignment[0],
                1: question_assignment[1],
                2: question_assignment[2],
            }
        )

        non_practice = non_practice[
            non_practice["onestopqa_question_id"] == selected_question
        ].copy()

        non_practice["question_list"] = question_list_name
        practice["question_list"] = question_list_name

        question_version_out = pd.concat(
            [non_practice, practice],
            ignore_index=True
        )

        question_version_out["final_experiment_list"] = (
            question_version_out["experiment_list"].astype(str)
            + "_"
            + question_list_name
        )

        all_parts.append(question_version_out)

    out = pd.concat(all_parts, ignore_index=True)

    return out


def build_regime_order_assignment_table(experiment_lists_df, seed=42):
    """
    Builds the 54-list assignment table.

    For each regime list letter (list_A / list_B / list_C):
    - take q0, q1, q2 twice
    - randomly assign O1-O6 to them

    The q-version <-> ordering pairing is drawn ONCE per list letter and reused
    in every batch, so a given (list letter, ordering) always means the same
    question version and the same regime order in all three batches.

    Across:
    3 batches x 3 regime lists x 6 orderings = 54 final lists.
    """

    rng = random.Random(seed)

    base_lists = sorted(experiment_lists_df["final_experiment_list"].unique())

    # Extract the base regime-list name by removing trailing _q0/_q1/_q2
    base_regime_lists = sorted(
        set(
            name.rsplit("_q", 1)[0]
            for name in base_lists
        )
    )

    # e.g. "batch_1_list_A" -> "list_A"
    list_letters = sorted(
        set(name.rsplit("_list_", 1)[-1] for name in base_regime_lists)
    )

    order_labels_by_letter = {}
    for letter in list_letters:
        order_labels = list(REGIME_ORDERINGS.keys())
        rng.shuffle(order_labels)
        order_labels_by_letter[letter] = order_labels

    rows = []

    for base_regime_list in base_regime_lists:

        q_versions = [
            f"{base_regime_list}_q0",
            f"{base_regime_list}_q1",
            f"{base_regime_list}_q2",
            f"{base_regime_list}_q0",
            f"{base_regime_list}_q1",
            f"{base_regime_list}_q2",
        ]

        letter = base_regime_list.rsplit("_list_", 1)[-1]
        order_labels = order_labels_by_letter[letter]

        for i, (source_list, order_label) in enumerate(zip(q_versions, order_labels), start=1):

            rows.append({
                "source_experiment_list": source_list,
                "order_label": order_label,
                "regime_order": REGIME_ORDERINGS[order_label],
                "ordered_experiment_list": f"{source_list}_{order_label}",
                "within_base_list_copy": i,
            })

    assignment_table = pd.DataFrame(rows)

    return assignment_table


def apply_regime_order_assignments(experiment_lists_df, assignment_table):
    """
    Expands the current 27-list dataframe into 54 final ordered lists.

    Keeps all original columns from experiment_lists_df and only adds:
    - ordered_experiment_list
    - order_label
    - regime_order
    - regime_order_position
    """

    all_parts = []

    for _, assignment in assignment_table.iterrows():

        source_list = assignment["source_experiment_list"]
        ordered_list = assignment["ordered_experiment_list"]
        order_label = assignment["order_label"]
        regime_order = assignment["regime_order"]

        # IMPORTANT:
        # This keeps ALL columns from the original experiment_lists_df
        df_part = experiment_lists_df[
            experiment_lists_df["final_experiment_list"] == source_list
        ].copy()

        # Add ordering metadata
        df_part["ordered_experiment_list"] = ordered_list
        df_part["order_label"] = order_label
        df_part["regime_order"] = " -> ".join(regime_order)

        regime_position_map = {
            regime: position
            for position, regime in enumerate(regime_order, start=1)
        }

        df_part["regime_order_position"] = df_part["regime"].map(regime_position_map)

        all_parts.append(df_part)

    ordered_experiment_lists_df = pd.concat(all_parts, ignore_index=True)

    return ordered_experiment_lists_df


def keep_one_practice_question_per_regime_block(df):
    """
    For each final ordered list:
    - keep all non-practice rows
    - for practice rows, keep only one practice paragraph per regime block:
        block 1 gets the first practice paragraph
        block 2 gets the second practice paragraph
        block 3 gets the third practice paragraph

    This uses sorted practice paragraph_id values, so it works whether
    they are 0,1,2 or 1,2,3.
    """

    df = df.copy()

    all_parts = []

    for list_name, list_df in df.groupby("ordered_experiment_list"):

        non_practice = list_df[list_df["practice"] == 0].copy()
        practice = list_df[list_df["practice"] == 1].copy()

        # Get the available practice paragraphs in order
        practice_paragraphs = sorted(practice["paragraph_id"].unique())

        if len(practice_paragraphs) < 3:
            raise ValueError(
                f"{list_name} has fewer than 3 practice paragraphs: "
                f"{practice_paragraphs}"
            )

        # Map regime block position to practice paragraph
        # block 1 -> first practice paragraph
        # block 2 -> second practice paragraph
        # block 3 -> third practice paragraph
        block_to_practice_paragraph = {
            1: practice_paragraphs[0],
            2: practice_paragraphs[1],
            3: practice_paragraphs[2],
        }

        practice["target_practice_paragraph"] = practice["regime_order_position"].map(
            block_to_practice_paragraph
        )

        practice = practice[
            practice["paragraph_id"] == practice["target_practice_paragraph"]
        ].copy()

        practice = practice.drop(columns=["target_practice_paragraph"])

        all_parts.append(pd.concat([non_practice, practice], ignore_index=True))

    out = pd.concat(all_parts, ignore_index=True)

    return out


def randomize_answers(row):
    order = rng.permutation(4)  # e.g. [1, 0, 3, 2]

    # answers shown on screen in randomized order
    reordered_answers = {
        f"answer_{screen_pos}": row[answer_cols[original_answer_idx]]
        for screen_pos, original_answer_idx in enumerate(order)
    }

    # keys: where each original answer moved to
    # a_key = position of answer_A, b_key = position of answer_B, etc.
    inverse_order = np.empty(4, dtype=int)
    inverse_order[order] = np.arange(4)

    keys = {
        "a_key": inverse_order[0],
        "b_key": inverse_order[1],
        "c_key": inverse_order[2],
        "d_key": inverse_order[3],
    }

    return pd.Series({
        **reordered_answers,
        "correct_answer": inverse_order[0],   # where answer_A moved
        **keys,
        "answers_order": [int(x) for x in order],
    })


def make_small_demo_from_batch(
    batch_df,
    n_articles_per_regime=2,
    n_paragraphs_per_article=2,
):
    """
    Create a small demo version from one batch.

    For each ordered_experiment_list and each regime:
    - keep the one practice trial for that regime
    - keep only the first n_articles_per_regime experimental articles
    - within those articles, keep only the first n_paragraphs_per_article paragraphs

    Assumes practice rows are already reduced to one practice row per regime block.
    """

    demo_parts = []

    for list_name, list_df in batch_df.groupby("ordered_experiment_list"):

        for regime, regime_df in list_df.groupby("regime"):

            # Keep practice trial for this regime
            practice_rows = regime_df[regime_df["practice"] == 1].copy()

            # Experimental rows only
            experimental_rows = regime_df[regime_df["practice"] == 0].copy()

            # Pick first N experimental articles in this regime
            selected_articles = sorted(experimental_rows["article_id"].unique())[
                :n_articles_per_regime
            ]

            experimental_rows = experimental_rows[
                experimental_rows["article_id"].isin(selected_articles)
            ].copy()

            # For each selected article, keep first N paragraphs
            keep_parts = []

            for article_id, article_df in experimental_rows.groupby("article_id"):

                selected_paragraphs = sorted(article_df["paragraph_id"].unique())[
                    :n_paragraphs_per_article
                ]

                article_keep = article_df[
                    article_df["paragraph_id"].isin(selected_paragraphs)
                ].copy()

                keep_parts.append(article_keep)

            if keep_parts:
                experimental_keep = pd.concat(keep_parts, ignore_index=True)
            else:
                experimental_keep = experimental_rows.iloc[0:0].copy()

            demo_parts.append(
                pd.concat([practice_rows, experimental_keep], ignore_index=True)
            )

    small_demo_df = pd.concat(demo_parts, ignore_index=True)

    return small_demo_df
