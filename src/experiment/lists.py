"""The Study 2 trial lists: which texts, questions and regime order each participant gets.

The texts come in three batches. Within each batch:

* **Regimes.** Articles 1-9 form three groups of three. A Latin square gives three regime
  lists (A, B, C), each giving one article group to each knowledge regime (full, partial,
  no knowledge). Article 10 is the practice article; article 0 is not used.
* **Questions.** Each paragraph has three questions, and a trial asks one of them. A second
  Latin square gives three question versions (q0, q1, q2): which question a paragraph gets
  depends on its position in the article, shifted by one from version to version. That
  makes 9 lists per batch, 27 in all.
* **Regime order.** Each regime list is given all six orders of the three regimes, two per
  question version: 18 ordered lists per batch, 54 in all. Which order goes with which
  question version is drawn once (seed 42) and is the same in every batch.
* **Practice and answers.** Each regime block gets one practice trial, and every trial's
  four answers are shuffled.

Each ordered list has 50-53 trials, 3 of them practice. The functions are listed in the
order they are run. `notebooks/experiment/lists_builder.ipynb` ran them and saved the lists
to `data/stimuli/new_batched_lists/`, for the presentation software (not in this repo).

**Reproducing the lists.** The answer shuffle draws from one generator, `rng`, seeded when
the module is imported, one trial at a time. So it repeats only for the same rows in the
same order, from a fresh import: adding, removing or reordering a row changes the answer
order of every trial after it.
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

# The experimental articles of a batch, in the three groups the regimes are given to.
ARTICLE_GROUPS = {
    "group_1": [1, 2, 3],
    "group_2": [4, 5, 6],
    "group_3": [7, 8, 9],
}

# Row k is regime list k: the regime of group_1, group_2 and group_3.
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

# For each question version, the question (0-2) asked about paragraphs 1, 2 and 3 of an
# article; paragraphs 4-6, 7-9, ... repeat the pattern.
QUESTION_LATIN_SQUARE = {
    "q0": [0, 1, 2],
    "q1": [1, 2, 0],
    "q2": [2, 0, 1],
}

REGIMES = ["full knowledge", "partial knowledge", "no knowledge"]

# The six orders of the three regimes, O1-O6.
REGIME_ORDERINGS = {
    f"O{i}": list(order)
    for i, order in enumerate(permutations(REGIMES), start=1)
}

# The answers by meaning; answer_A is the correct one.
answer_cols = ["answer_A", "answer_B", "answer_C", "answer_D"]

# The answer shuffle's generator, seeded once, when the module is imported.
rng = np.random.default_rng(seed=42)


def repair_mojibake(value):
    """Fix text garbled by an encoding mix-up (e.g. "â€™" for "’"), with `ftfy`.
    Anything that is not text comes back as it is."""
    if isinstance(value, str):
        return fix_text(value)
    return value


def clean_batch_articles(df):
    """Drop article 0, and mark article 10 as the practice article (`practice` = 1,
    else 0). An article id may be written as 0, article_0 or id_0."""
    df = df.copy()

    article_id_str = df["article_id"].astype(str).str.strip()
    df = df[~article_id_str.isin(["0", "article_0", "id_0"])].copy()

    article_id_str = df["article_id"].astype(str).str.strip()
    df["practice"] = article_id_str.isin(["10", "article_10", "id_10"]).astype(int)

    return df


def restrict_practice_questions(df):
    """Keep every experimental row, and of the practice article only question 0 of
    paragraphs 1, 2 and 3: the three practice trials."""
    df = df.copy()

    # As numbers, in case the columns were read as text.
    paragraph_num = pd.to_numeric(df["paragraph_id"], errors="coerce")
    question_num = pd.to_numeric(df["onestopqa_question_id"], errors="coerce")

    non_practice_mask = df["practice"] == 0
    practice_keep_mask = (
        (df["practice"] == 1)
        & (paragraph_num.isin([1, 2, 3]))
        & (question_num == 0)
    )

    return df[non_practice_mask | practice_keep_mask].copy()


def build_latin_square_lists_for_batch(df, batch_name):
    """The three regime lists of one batch (A, B, C).

    In each list, every article group gets the regime `LATIN_SQUARE` gives it, and the
    practice trials are copied into all three regimes. Adds `experiment_list` (e.g.
    "batch_1_list_A"), `regime_list`, `regime` and `article_group` ("practice" for the
    practice trials).
    """

    df = df.copy()

    experimental_rows = df[df["practice"] == 0].copy()
    practice_rows = df[df["practice"] == 1].copy()

    all_list_parts = []

    for list_idx, regime_assignment in enumerate(LATIN_SQUARE, start=1):

        list_name = LIST_NAMES[list_idx]

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
    """Make the three question versions (q0, q1, q2) of every regime list.

    In each version an experimental paragraph keeps one of its three questions, the one
    `QUESTION_LATIN_SQUARE` gives its position in the article. The practice trials are
    copied into every version unchanged. Adds `question_list` and
    `final_experiment_list` (e.g. "batch_1_list_A_q0").
    """

    all_parts = []

    for question_list_name, question_assignment in QUESTION_LATIN_SQUARE.items():

        question_version = df.copy()

        non_practice = question_version[question_version["practice"] == 0].copy()
        practice = question_version[question_version["practice"] == 1].copy()

        # Paragraphs 1, 2, 3, 4, ... are in positions 0, 1, 2, 0, ...
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
    """Which regime order each list gets: one row per ordered list, 54 in all.

    For each regime-list letter (A, B, C), the six orders O1-O6 are shuffled once, with
    `seed`, and paired with q0, q1, q2, q0, q1, q2. The same pairing is used in every
    batch, so a list letter and an order always mean the same question version and the
    same regime order. Columns: `source_experiment_list`, `order_label`, `regime_order`
    (the three regimes, in order), `ordered_experiment_list` and `within_base_list_copy`.
    """

    rng = random.Random(seed)

    base_lists = sorted(experiment_lists_df["final_experiment_list"].unique())

    # The regime list each question version belongs to: the name without _q0/_q1/_q2.
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
    """Make the 54 ordered lists from the 27 lists: one copy of a list for each row of
    `assignment_table` that names it.

    Every column is kept, and four are added: `ordered_experiment_list`, `order_label`,
    `regime_order` (as "full knowledge -> no knowledge -> ...") and
    `regime_order_position` (1-3: where the trial's regime comes in that order).
    """

    all_parts = []

    for _, assignment in assignment_table.iterrows():

        source_list = assignment["source_experiment_list"]
        ordered_list = assignment["ordered_experiment_list"]
        order_label = assignment["order_label"]
        regime_order = assignment["regime_order"]

        df_part = experiment_lists_df[
            experiment_lists_df["final_experiment_list"] == source_list
        ].copy()

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
    """Keep one practice trial in each regime block of every ordered list.

    The first regime in the list's order gets the first practice paragraph, the second
    the second, the third the third (by `paragraph_id`, sorted). Experimental trials are
    all kept. Stops with an error if a list has fewer than three practice paragraphs.
    """

    df = df.copy()

    all_parts = []

    for list_name, list_df in df.groupby("ordered_experiment_list"):

        non_practice = list_df[list_df["practice"] == 0].copy()
        practice = list_df[list_df["practice"] == 1].copy()

        practice_paragraphs = sorted(practice["paragraph_id"].unique())

        if len(practice_paragraphs) < 3:
            raise ValueError(
                f"{list_name} has fewer than 3 practice paragraphs: "
                f"{practice_paragraphs}"
            )

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
    """Shuffle one trial's four answers into a random screen order.

    Returns the answers by screen position (`answer_0` .. `answer_3`); `correct_answer`,
    the position of answer_A (the correct one); `a_key` .. `d_key`, the position of each
    answer; and `answers_order`, which answer is at each position (0 = A ... 3 = D).
    Draws from the module's `rng`, so the result depends on every trial shuffled before
    it (see the module docstring).
    """
    order = rng.permutation(4)  # e.g. [1, 0, 3, 2]: answer_B first, then answer_A, ...

    reordered_answers = {
        f"answer_{screen_pos}": row[answer_cols[original_answer_idx]]
        for screen_pos, original_answer_idx in enumerate(order)
    }

    # Where each answer went: a_key is the position of answer_A, b_key of answer_B, ...
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
        "correct_answer": inverse_order[0],
        **keys,
        "answers_order": [int(x) for x in order],
    })


def make_small_demo_from_batch(
    batch_df,
    n_articles_per_regime=2,
    n_paragraphs_per_article=2,
):
    """A short version of one batch, for trying the experiment out.

    In every ordered list and regime: the practice trial, plus the first
    `n_paragraphs_per_article` paragraphs of the first `n_articles_per_regime` articles.
    Expects the practice trials already cut to one per regime block
    (`keep_one_practice_question_per_regime_block`).
    """

    demo_parts = []

    for list_name, list_df in batch_df.groupby("ordered_experiment_list"):

        for regime, regime_df in list_df.groupby("regime"):

            practice_rows = regime_df[regime_df["practice"] == 1].copy()
            experimental_rows = regime_df[regime_df["practice"] == 0].copy()

            selected_articles = sorted(experimental_rows["article_id"].unique())[
                :n_articles_per_regime
            ]

            experimental_rows = experimental_rows[
                experimental_rows["article_id"].isin(selected_articles)
            ].copy()

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
