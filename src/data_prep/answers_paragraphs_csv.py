"""
answers_paragraphs_csv.py

Builds wide CSVs that combine:
- answer-level dwell-time features (per area, plus selected answer)
  from the hunters/gatherers CSVs produced by data_csv_generation.py
- paragraph-level dwell-time features (auxiliary span types) from ia_P

Expected inputs
---------------
1) Answers feature CSVs (already processed by data_csv_generation.main):
   - hunters_answers_path
   - gatherers_answers_path
   These files must contain (among others):
       TRIAL_ID, PARTICIPANT_ID,
       TEXT_ID_WITH_Q_COLUMN,
       AREA_LABEL_COLUMN,
       AREA_SCREEN_LOCATION,
       MEAN_DWELL_TIME,
       SELECTED_ANSWER_LABEL_COLUMN,
       SELECTED_ANSWER_POSITION_COLUMN.

2) Raw paragraphs IA CSV: ia_paragraphs_path (usually "full/ia_P.csv")

Outputs
-------
- data/merged_hunters.csv
- data/merged_gatherers.csv
"""

import os
import pandas as pd

from src import constants as C
from src.data_prep.data_csv_generation import (
    load_raw_paragraphs_data,
    split_hunters_and_gatherers,
)


# ---------------------------------------------------------------------------
# Answer-side helpers
# ---------------------------------------------------------------------------

def load_answer_features(
    hunters_answers_path: str,
    gatherers_answers_path: str,
):
    """
    Load hunters and gatherers answers CSVs that were produced by
    data_csv_generation.main().

    Returns
    -------
    df_hunters, df_gatherers : DataFrame, DataFrame
    """
    print(f"Loading processed hunters answers from: {hunters_answers_path}")
    df_h = pd.read_csv(hunters_answers_path)

    print(f"Loading processed gatherers answers from: {gatherers_answers_path}")
    df_g = pd.read_csv(gatherers_answers_path)

    return df_h, df_g


def create_mean_area_dwell_time_answers_from_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct an area-level answers table from the processed IA CSV.

    The input is the *row-level* IA data with group features already merged
    in by data_csv_generation.generate_new_row_features(). We assume that:
    - C.MEAN_DWELL_TIME is already present and constant per
      (TRIAL_ID, PARTICIPANT_ID, TEXT_ID_WITH_Q_COLUMN, AREA_LABEL_COLUMN).
    - C.AREA_SCREEN_LOCATION,
      C.SELECTED_ANSWER_LABEL_COLUMN,
      C.SELECTED_ANSWER_POSITION_COLUMN
      are also present.

    We simply select the relevant columns and drop duplicates to get
    one row per area per trial/participant/text.
    """
    df = df.copy()

    cols = [
        C.TRIAL_ID,
        C.PARTICIPANT_ID,
        C.TEXT_ID_WITH_Q_COLUMN,
        C.AREA_LABEL_COLUMN,
        C.AREA_SCREEN_LOCATION,
        C.MEAN_DWELL_TIME,
        C.SELECTED_ANSWER_LABEL_COLUMN,
        C.SELECTED_ANSWER_POSITION_COLUMN,
    ]

    # One row per (trial, participant, text, area_label)
    key_cols = [
        C.TRIAL_ID,
        C.PARTICIPANT_ID,
        C.TEXT_ID_WITH_Q_COLUMN,
        C.AREA_LABEL_COLUMN,
    ]

    area_df = df[cols].drop_duplicates(subset=key_cols)
    return area_df


def pivot_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide-format answer-level table.

    Uses TEXT_ID_WITH_Q_COLUMN as the text identifier,
    matching the original text_id logic.
    """
    idx_cols = [C.TRIAL_ID, C.PARTICIPANT_ID, C.TEXT_ID_WITH_Q_COLUMN]

    dwell_wide = (
        df.pivot(
            index=idx_cols,
            columns=C.AREA_LABEL_COLUMN,
            values=C.MEAN_DWELL_TIME,
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )

    loc_wide = (
        df.pivot(
            index=idx_cols,
            columns=C.AREA_LABEL_COLUMN,
            values=C.AREA_SCREEN_LOCATION,
        )
        .add_prefix("loc_")
        .rename_axis(None, axis=1)
        .reset_index()
    )

    sel = (
        df[
            idx_cols
            + [
                C.SELECTED_ANSWER_LABEL_COLUMN,
                C.SELECTED_ANSWER_POSITION_COLUMN,
            ]
        ]
        .drop_duplicates()
    )

    ans_wide = (
        dwell_wide
        .merge(loc_wide, on=idx_cols)
        .merge(sel, on=idx_cols)
    )

    def _get_dwell_selected(row):
        label = row[C.SELECTED_ANSWER_LABEL_COLUMN]
        col_name = f"answer_{label}"
        return row.get(col_name, None)

    def _get_loc_selected(row):
        label = row[C.SELECTED_ANSWER_LABEL_COLUMN]
        col_name = f"loc_answer_{label}"
        return row.get(col_name, None)

    ans_wide["dwell_selected"] = ans_wide.apply(_get_dwell_selected, axis=1)
    ans_wide["screenloc_selected"] = ans_wide.apply(_get_loc_selected, axis=1)

    return ans_wide


# ---------------------------------------------------------------------------
# Paragraph / text-side helpers
# ---------------------------------------------------------------------------

def create_mean_area_dwell_time_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean dwell time per (trial, participant, auxiliary_span_type) for paragraphs.
    """
    df = df.copy()
    return (
        df.groupby(
            [C.TRIAL_ID, C.PARTICIPANT_ID, C.AUXILIARY_SPAN_TYPE_COLUMN],
            as_index=False,
        ).agg(**{C.MEAN_DWELL_TIME: (C.IA_DWELL_TIME, "mean")})
    )


def pivot_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide-format text-level table.

    Input: output of create_mean_area_dwell_time_text(df).
    Output: one row per (TRIAL_ID, PARTICIPANT_ID) with columns per AUXILIARY_SPAN_TYPE_COLUMN.
    """
    return (
        df.pivot(
            index=[C.TRIAL_ID, C.PARTICIPANT_ID],
            columns=C.AUXILIARY_SPAN_TYPE_COLUMN,
            values=C.MEAN_DWELL_TIME,
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Main script logic
# ---------------------------------------------------------------------------

def build_merged_tables(
    hunters_answers_path: str = "data/hunters.csv",
    gatherers_answers_path: str = "data/gatherers.csv",
    ia_paragraphs_path: str = "data_raw/full/ia_P.csv",
):
    """
    The full pipeline:

    1) Load processed answers feature CSVs for hunters / gatherers.
    2) From these, build area-level mean dwell tables and pivot to wide
       (answers side).
    3) Load raw ia_P, split into hunters / gatherers using
       split_hunters_and_gatherers().
    4) For paragraphs:
       - compute mean dwell times per auxiliary span type
       - pivot to wide format (texts)
    5) Merge answers + texts for hunters and gatherers separately.
    """


    df_A_h_processed, df_A_g_processed = load_answer_features(
        hunters_answers_path=hunters_answers_path,
        gatherers_answers_path=gatherers_answers_path,
    )

    print("Building area-level answers table (hunters)...")
    hunters_dwells_a = create_mean_area_dwell_time_answers_from_processed(
        df_A_h_processed
    )

    print("Building area-level answers table (gatherers)...")
    gatherers_dwells_a = create_mean_area_dwell_time_answers_from_processed(
        df_A_g_processed
    )

    print("Pivoting answers to wide format (hunters)...")
    ans_wide_h = pivot_answers(hunters_dwells_a)

    print("Pivoting answers to wide format (gatherers)...")
    ans_wide_g = pivot_answers(gatherers_dwells_a)

    print(f"Loading paragraphs from: {ia_paragraphs_path}")
    df_P = load_raw_paragraphs_data(ia_paragraphs_path)

    print("Splitting paragraphs into hunters and gatherers...")
    df_P_hunters, df_P_gatherers = split_hunters_and_gatherers(df_P)

    print("Aggregating text-level dwell times (hunters)...")
    hunters_dwells_p = create_mean_area_dwell_time_text(df_P_hunters)

    print("Aggregating text-level dwell times (gatherers)...")
    gatherers_dwells_p = create_mean_area_dwell_time_text(df_P_gatherers)

    print("Pivoting texts to wide format (hunters)...")
    text_wide_h = pivot_texts(hunters_dwells_p)

    print("Pivoting texts to wide format (gatherers)...")
    text_wide_g = pivot_texts(gatherers_dwells_p)

    print("Merging answer- and text-level tables (hunters)...")
    merged_h = text_wide_h.merge(
        ans_wide_h,
        on=[C.TRIAL_ID, C.PARTICIPANT_ID],
        how="inner",
    )

    print("Merging answer- and text-level tables (gatherers)...")
    merged_g = text_wide_g.merge(
        ans_wide_g,
        on=[C.TRIAL_ID, C.PARTICIPANT_ID],
        how="inner",
    )

    return merged_h, merged_g


def main(
    hunters_answers_path: str = "data/hunters.csv",
    gatherers_answers_path: str = "data/gatherers.csv",
    ia_paragraphs_path: str = "data_raw/full/ia_P.csv",
    hunters_output_path: str = "data/merged_hunters.csv",
    gatherers_output_path: str = "data/merged_gatherers.csv",
):
    """
    run the pipeline and save CSVs.
    """
    merged_h, merged_g = build_merged_tables(
        hunters_answers_path=hunters_answers_path,
        gatherers_answers_path=gatherers_answers_path,
        ia_paragraphs_path=ia_paragraphs_path,
    )

    os.makedirs(os.path.dirname(hunters_output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(gatherers_output_path) or ".", exist_ok=True)

    print(f"Saving hunters merged CSV to: {hunters_output_path}")
    merged_h.to_csv(hunters_output_path, index=False)

    print(f"Saving gatherers merged CSV to: {gatherers_output_path}")
    merged_g.to_csv(gatherers_output_path, index=False)

    print("✓ Done.")


if __name__ == "__main__":
    main()
