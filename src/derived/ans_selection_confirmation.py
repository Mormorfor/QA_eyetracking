from __future__ import annotations

import ast
from functools import reduce
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
import src.constants as Con


def truncate_recordings_at_first_malformed_trial(
    df: pd.DataFrame,
    recording_col: str = Con.RECORDING_SESSION_LABEL,
    trial_col: str = Con.TRIAL_ID,
    answers_col: str = Con.ALL_ANSWERS,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep valid trials at the beginning of each recording and drop all trials
    from the first malformed ALL_ANSWERS trial onward.
    """

    def safe_parse_answers(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except Exception:
            return None

    trial_answers = (
        df[[recording_col, trial_col, answers_col]]
        .drop_duplicates(subset=[recording_col, trial_col])
        .sort_values([recording_col, trial_col])
        .copy()
    )

    trial_answers[Con.ALL_ANSWERS_LIST] = trial_answers[answers_col].apply(safe_parse_answers)
    trial_answers[Con.IS_MALFORMED] = trial_answers[Con.ALL_ANSWERS_LIST].isna()

    bad_trial_rows = trial_answers.loc[trial_answers[Con.IS_MALFORMED]].copy()

    malformed_summary_df = (
        bad_trial_rows.groupby(recording_col, as_index=False)[trial_col]
        .min()
        .rename(columns={trial_col: Con.FIRST_MALFORMED_TRIAL})
        .sort_values(Con.FIRST_MALFORMED_TRIAL)
    )

    out = df.merge(malformed_summary_df, on=recording_col, how="left")

    keep_mask = (
        out[Con.FIRST_MALFORMED_TRIAL].isna()
        | (out[trial_col] < out[Con.FIRST_MALFORMED_TRIAL])
    )

    cleaned_df = out.loc[keep_mask].copy().drop(columns=[Con.FIRST_MALFORMED_TRIAL])

    if verbose:
        n_bad_recordings = malformed_summary_df[recording_col].nunique()
        print(f"Malformed trial rows: {len(bad_trial_rows)}")
        print(f"Affected recordings: {n_bad_recordings}")
        print(f"Rows: {len(df)} -> {len(cleaned_df)}")
        if n_bad_recordings > 0:
            print("\nFirst malformed trial per affected recording:")
            print(malformed_summary_df.head(10))

    return cleaned_df, malformed_summary_df, bad_trial_rows



def build_trial_answers_df(
    df: pd.DataFrame,
    participant_col: str = Con.RECORDING_SESSION_LABEL,
    trial_col: str = Con.TRIAL_ID,
    answers_col: str = Con.ALL_ANSWERS,
) -> pd.DataFrame:
    """
    Build a one-row-per-participant/trial dataframe with cumulative answers,
    previous cumulative answers, and newly added trial answers.
    """
    trial_answers = (
        df[[participant_col, trial_col, answers_col]]
        .drop_duplicates(subset=[participant_col, trial_col])
        .sort_values([participant_col, trial_col])
        .copy()
    )

    trial_answers[Con.ALL_ANSWERS_LIST] = trial_answers[answers_col].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )

    trial_answers[Con.PREV_ALL_ANSWERS_LIST] = (
        trial_answers.groupby(participant_col)[Con.ALL_ANSWERS_LIST].shift(1)
    )

    trial_answers[Con.TRIAL_ANSWERS] = trial_answers[Con.ALL_ANSWERS_LIST]

    mask = trial_answers[Con.PREV_ALL_ANSWERS_LIST].notna()
    trial_answers.loc[mask, Con.TRIAL_ANSWERS] = trial_answers.loc[mask].apply(
        lambda row: row[Con.ALL_ANSWERS_LIST][len(row[Con.PREV_ALL_ANSWERS_LIST]):],
        axis=1,
    )

    return trial_answers



def extract_click_timestamps(
    df: pd.DataFrame,
    participant_col: str = Con.RECORDING_SESSION_LABEL,
    trial_col: str = Con.TRIAL_ID,
    keyword: str = Con.CHOOSE_ANSWER_KEYWORD,
    text_list_col_1: str = Con.CURRENT_FIX_MSG_LIST_TEXT,
    time_list_col_1: str = Con.CURRENT_FIX_MSG_LIST_TIME,
    text_list_col_2: str = Con.NEXT_SAC_MSG_LIST_TEXT,
    time_list_col_2: str = Con.NEXT_SAC_MSG_LIST_TIME,
    output_col: str = Con.SELECT_ANS_TIMESTAMPS,
) -> pd.DataFrame:
    """
    Collect timestamps whose paired message text contains `keyword`
    from two list-based message sources, combine them, sort them,
    and deduplicate them within each participant/trial.
    """

    keys = [participant_col, trial_col]

    def parse_text_list(x):
        if x == Con.MISSING_LIST_MARKER:
            return []
        inner = x[1:-1]
        return [] if inner == "" else inner.split(", ")

    def parse_time_list(x):
        if x == Con.MISSING_LIST_MARKER:
            return []
        inner = x[1:-1]
        return [] if inner == "" else [int(t) for t in inner.split(", ")]

    def extract_matching_timestamps(texts, times, keyword):
        return [t for txt, t in zip(texts, times) if keyword in txt]

    part = df[
        keys + [text_list_col_1, time_list_col_1, text_list_col_2, time_list_col_2]
    ].copy()

    part[text_list_col_1] = part[text_list_col_1].apply(parse_text_list)
    part[text_list_col_2] = part[text_list_col_2].apply(parse_text_list)
    part[time_list_col_1] = part[time_list_col_1].apply(parse_time_list)
    part[time_list_col_2] = part[time_list_col_2].apply(parse_time_list)

    part[Con.MATCH_TIMESTAMPS_1] = part.apply(
        lambda row: extract_matching_timestamps(
            row[text_list_col_1], row[time_list_col_1], keyword
        ),
        axis=1,
    )
    part[Con.MATCH_TIMESTAMPS_2] = part.apply(
        lambda row: extract_matching_timestamps(
            row[text_list_col_2], row[time_list_col_2], keyword
        ),
        axis=1,
    )

    part[Con.MATCH_TIMESTAMPS] = part[Con.MATCH_TIMESTAMPS_1] + part[Con.MATCH_TIMESTAMPS_2]
    part = part[keys + [Con.MATCH_TIMESTAMPS]].explode(Con.MATCH_TIMESTAMPS)
    part = part.rename(columns={Con.MATCH_TIMESTAMPS: Con.MATCH_TIMESTAMP})

    part[Con.MATCH_TIMESTAMP] = pd.to_numeric(part[Con.MATCH_TIMESTAMP], errors="coerce")
    part = part.dropna(subset=[Con.MATCH_TIMESTAMP]).copy()
    part[Con.MATCH_TIMESTAMP] = part[Con.MATCH_TIMESTAMP].astype(int)

    grouped = (
        part.groupby(keys)[Con.MATCH_TIMESTAMP]
        .apply(lambda x: list(dict.fromkeys(sorted(x.tolist()))))
        .reset_index()
        .rename(columns={Con.MATCH_TIMESTAMP: output_col})
    )

    return grouped



def extract_fixation_timestamps_with_ia(
    df: pd.DataFrame,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
    label_col: str = Con.CURRENT_FIX_LABEL,
    ia_col: str = Con.NEAREST_IA,
) -> pd.DataFrame:
    """
    Extract fixation timestamps and corresponding interest area values,
    returning a list of (timestamp, ia) tuples per participant/trial.
    """
    out = df[[participant_col, trial_col, label_col, ia_col]].copy()

    out[Con.FIXATION_TIMESTAMP] = (
        out[label_col].astype(str).str.extract(r"Fixation:\s*(\d+)\s*ms", expand=False)
    )
    out[Con.FIXATION_TIMESTAMP] = pd.to_numeric(out[Con.FIXATION_TIMESTAMP], errors="coerce")
    out = out.dropna(subset=[Con.FIXATION_TIMESTAMP]).copy()
    out[Con.FIXATION_TIMESTAMP] = out[Con.FIXATION_TIMESTAMP].astype(int)

    out[Con.FIXATION_PAIR] = list(zip(out[Con.FIXATION_TIMESTAMP], out[ia_col]))

    fixation_df = (
        out.groupby([participant_col, trial_col])[Con.FIXATION_PAIR]
        .apply(list)
        .reset_index()
        .rename(columns={Con.FIXATION_PAIR: Con.FIXATION_TIMESTAMPS_IA})
    )

    return fixation_df



def join_on_participant_trial(
    dfs: list[pd.DataFrame],
    participant_col: str = Con.RECORDING_SESSION_LABEL,
    trial_col: str = Con.TRIAL_ID,
    how: str = "outer",
    sort: bool = True,
    participant_aliases: list[str] | None = None,
    trial_aliases: list[str] | None = None,
) -> pd.DataFrame:
    """
    Join multiple dataframes on participant + trial columns.
    Automatically renames alternative column names if found.
    """

    if participant_aliases is None:
        participant_aliases = [Con.PARTICIPANT_ID, Con.RECORDING_SESSION_LABEL]
    if trial_aliases is None:
        trial_aliases = [Con.TRIAL_ID]

    keys = [participant_col, trial_col]
    normalized_dfs = []

    for df in dfs:
        cur = df.copy()

        if participant_col not in cur.columns:
            for alias in participant_aliases:
                if alias in cur.columns:
                    cur = cur.rename(columns={alias: participant_col})
                    break

        if trial_col not in cur.columns:
            for alias in trial_aliases:
                if alias in cur.columns:
                    cur = cur.rename(columns={alias: trial_col})
                    break

        normalized_dfs.append(cur)

    merged = reduce(
        lambda left, right: pd.merge(left, right, on=keys, how=how),
        normalized_dfs,
    )

    if sort:
        merged = merged.sort_values(keys).reset_index(drop=True)

    return merged



def extract_last_fixations_before_clicks(
    df: pd.DataFrame,
    fixation_col: str = Con.FIXATION_TIMESTAMPS_IA,
    click_col: str = Con.SELECT_ANS_TIMESTAMPS,
    output_col: str = Con.LAST_FIXATIONS_BEFORE_SELECT,
) -> pd.DataFrame:
    """
    For each click timestamp in a row, find the last fixation tuple
    (timestamp, ia) occurring at or before that click.
    """

    def last_fixations_for_row(fixations, clicks):
        if not isinstance(fixations, list) or not isinstance(clicks, list):
            return []

        clean_fixations = []
        for f in fixations:
            if isinstance(f, tuple) and len(f) == 2:
                t = int(f[0])
                ia = f[1]
                clean_fixations.append((t, ia))

        clean_clicks = [int(c) for c in clicks if c is not None]

        if not clean_fixations or not clean_clicks:
            return []

        clean_fixations = sorted(clean_fixations, key=lambda x: x[0])

        result = []
        i = 0
        last_seen = None

        for click in clean_clicks:
            while i < len(clean_fixations) and clean_fixations[i][0] <= click:
                last_seen = clean_fixations[i]
                i += 1

            if last_seen is not None:
                result.append(last_seen)

        return result

    out = df.copy()
    out[output_col] = out.apply(
        lambda row: last_fixations_for_row(row[fixation_col], row[click_col]),
        axis=1,
    )
    return out



def build_trial_level_df(
    fix_tsv: pd.DataFrame,
    fix_csv: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run the full trial-level pipeline and return the final dataframe together
    with useful intermediate diagnostics.
    """
    fix_tsv_cleaned, malformed_summary_df, bad_trial_rows = (
        truncate_recordings_at_first_malformed_trial(fix_tsv, verbose=verbose)
    )

    trial_answers = build_trial_answers_df(fix_tsv_cleaned)

    click_df = extract_click_timestamps(
        fix_tsv_cleaned,
        keyword=Con.CHOOSE_ANSWER_KEYWORD,
        output_col=Con.SELECT_ANS_TIMESTAMPS,
    )

    confirm_df = extract_click_timestamps(
        fix_tsv_cleaned,
        keyword=Con.CONFIRM_ANSWER_KEYWORD,
        output_col=Con.CONFIRM_TIMESTAMPS,
    )

    fixation_df = extract_fixation_timestamps_with_ia(fix_csv)

    trial_level_df = join_on_participant_trial(
        [trial_answers, click_df, fixation_df, confirm_df]
    )

    trial_level_df = extract_last_fixations_before_clicks(
        trial_level_df,
        fixation_col=Con.FIXATION_TIMESTAMPS_IA,
        click_col=Con.SELECT_ANS_TIMESTAMPS,
        output_col=Con.LAST_FIXATIONS_BEFORE_SELECT,
    )
    trial_level_df = extract_last_fixations_before_clicks(
        trial_level_df,
        fixation_col=Con.FIXATION_TIMESTAMPS_IA,
        click_col=Con.CONFIRM_TIMESTAMPS,
        output_col=Con.LAST_FIXATIONS_BEFORE_CONFIRM,
    )

    diagnostics = {
        "malformed_summary_df": malformed_summary_df,
        "bad_trial_rows": bad_trial_rows,
        "trial_answers": trial_answers,
        "click_df": click_df,
        "confirm_df": confirm_df,
        "fixation_df": fixation_df,
    }

    return trial_level_df, diagnostics



def run_trial_level_pipeline(
    fix_csv_path: str | Path = "../data_raw/full/fixations_A.csv",
    fix_tsv_path: str | Path = "../data_raw/unused/Fixations reports/fixations_A.tsv",
    output_csv_path: str | Path = "trial_level_df.csv",
    fix_tsv_encoding: str = "utf-16",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load inputs, run the full pipeline, save the final trial-level dataframe,
    and return it.
    """
    fix_csv_path = Path(fix_csv_path)
    fix_tsv_path = Path(fix_tsv_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading fix_csv from: {fix_csv_path}")
    fix_csv = pd.read_csv(fix_csv_path)

    if verbose:
        print(f"Loading fix_tsv from: {fix_tsv_path}")
    fix_tsv = pd.read_csv(fix_tsv_path, sep="\t", encoding=fix_tsv_encoding)

    trial_level_df, diagnostics = build_trial_level_df(
        fix_tsv=fix_tsv,
        fix_csv=fix_csv,
        verbose=verbose,
    )

    trial_level_df.to_csv(output_csv_path, index=False)

    if verbose:
        print(f"Saved trial_level_df to: {output_csv_path}")
        print(f"Final shape: {trial_level_df.shape}")

    return trial_level_df


if __name__ == "__main__":
    run_trial_level_pipeline()