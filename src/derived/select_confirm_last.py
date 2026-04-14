import ast
import pandas as pd


def _parse_tuple_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)


def _extract_last_ia_from_tuple_list(x):
    """
    From a list like:
    [(42231, 24), (44993, 9), (45873, 45), (46380, 32)]

    return:
    32

    Returns pd.NA for empty or malformed values.
    """

    try:
        tuples_ = _parse_tuple_list(x)
        if not tuples_:
            return pd.NA

        last_item = tuples_[-1]

        if isinstance(last_item, tuple) and len(last_item) == 2:
            ia = last_item[1]
            if pd.isna(ia):
                return pd.NA
            return int(ia)

        return pd.NA
    except Exception:
        return pd.NA


    # tuples_ = ast.literal_eval(x)
    # last_item = tuples_[-1]
    # ia = last_item[1]
    #
    # return int(ia)


def extract_last_areas_from_trial_level_df(
    trial_df: pd.DataFrame,
    participant_col: str = "participant_id",
    trial_col: str = "TRIAL_INDEX",
    select_fix_col: str = "LAST_FIXATIONS_BEFORE_SELECT",
    confirm_fix_col: str = "LAST_FIXATIONS_BEFORE_CONFIRM",
) -> pd.DataFrame:
    """
    Extract:
    - last IA before the last select
    - last IA before confirm

    directly from tuple-list columns in the trial-level CSV.
    """
    out = trial_df[
        [participant_col, trial_col, select_fix_col, confirm_fix_col]
    ].copy()

    out["last_ia_before_last_select"] = out[select_fix_col].apply(
        _extract_last_ia_from_tuple_list
    )

    out["last_ia_before_confirm"] = out[confirm_fix_col].apply(
        _extract_last_ia_from_tuple_list
    )

    return out[
        [
            participant_col,
            trial_col,
            "last_ia_before_last_select",
            "last_ia_before_confirm",
        ]
    ]


def attach_area_labels(
    extracted_df: pd.DataFrame,
    ia_df: pd.DataFrame,
    participant_col: str = "participant_id",
    trial_col: str = "TRIAL_INDEX",
    ia_id_col: str = "IA_ID",
    area_label_col: str = "area_label",
) -> pd.DataFrame:
    """
    Keep only participant/trial pairs that exist in ia_df.
    Then, within each valid participant/trial pair, attach the area label
    corresponding to:
    - last_ia_before_last_select
    - last_ia_before_confirm
    """

    ia_small = ia_df[
        [participant_col, trial_col, ia_id_col, area_label_col]
    ].copy()

    ia_small[ia_id_col] = pd.to_numeric(ia_small[ia_id_col], errors="coerce").astype("Int64")

    valid_pairs = ia_small[[participant_col, trial_col]].drop_duplicates()

    extracted = extracted_df.copy()
    extracted["last_ia_before_last_select"] = pd.to_numeric(
        extracted["last_ia_before_last_select"], errors="coerce"
    ).astype("Int64")

    extracted["last_ia_before_confirm"] = pd.to_numeric(
        extracted["last_ia_before_confirm"], errors="coerce"
    ).astype("Int64")

    # keep only participant/trial pairs that exist in the original ia_df
    extracted = extracted.merge(
        valid_pairs,
        on=[participant_col, trial_col],
        how="inner",
    )

    select_lookup = ia_small.rename(columns={
        ia_id_col: "last_ia_before_last_select",
        area_label_col: "area_label_before_last_select",
    })

    confirm_lookup = ia_small.rename(columns={
        ia_id_col: "last_ia_before_confirm",
        area_label_col: "area_label_before_confirm",
    })

    out = extracted.merge(
        select_lookup,
        on=[participant_col, trial_col, "last_ia_before_last_select"],
        how="left",
    )

    out = out.merge(
        confirm_lookup,
        on=[participant_col, trial_col, "last_ia_before_confirm"],
        how="left",
    )

    return out[
        [
            participant_col,
            trial_col,
            "last_ia_before_last_select",
            "area_label_before_last_select",
            "last_ia_before_confirm",
            "area_label_before_confirm",
        ]
    ]


def get_last_areas_from_trial_level_csv(
    trial_level_path="../data/select_confirm.csv",
    hunt_path="../data/hunters.csv",
    gath_path="../data/gatherers.csv",
    out_hunt_path="../data/hunters_last.csv",
    out_gath_path="../data/gatherers_last.csv",
    verbose=True,
):
    if verbose:
        print("Loading data...")

    trial_df = pd.read_csv(trial_level_path)
    hunt = pd.read_csv(hunt_path)
    gath = pd.read_csv(gath_path)

    if verbose:
        print("Extracting last IAs from trial-level data...")

    extracted = extract_last_areas_from_trial_level_df(trial_df)

    if verbose:
        print("Attaching area labels (hunters)...")

    fin_hunt = attach_area_labels(extracted, hunt)

    if verbose:
        print("Attaching area labels (gatherers)...")

    fin_gath = attach_area_labels(extracted, gath)

    if verbose:
        print("Saving outputs...")

    fin_hunt.to_csv(out_hunt_path, index=False)
    fin_gath.to_csv(out_gath_path, index=False)

    if verbose:
        print(f"Saved:\n- {out_hunt_path}\n- {out_gath_path}")

    return fin_hunt, fin_gath


if __name__ == "__main__":
    get_last_areas_from_trial_level_csv()