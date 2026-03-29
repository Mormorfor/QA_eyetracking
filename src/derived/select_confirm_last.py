from IPython.core.display_functions import display

from data_prep.data_csv_generation import load_raw_answers_data, load_raw_paragraphs_data
import pandas as pd


import pandas as pd
import re


def _extract_single_ia_value(x):
    """
    Convert values like '[ 31]' or '[31]' to 31.
    Treat '[0]' as empty.
    Return pd.NA for empty values like '[]', '[ ]', or '[0]'.
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    nums = re.findall(r'\d+', s)
    if not nums:
        return pd.NA

    val = int(nums[-1])

    # treat 0 as empty
    if val == 0:
        return pd.NA

    return val


def extract_last_ia_before_selection_and_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        [
            'participant_id',
            'TRIAL_INDEX',
            'CURRENT_FIX_END',
            'CURRENT_FIX_DURATION',
            'ANSWER_RT',
            'CONFIRM_FINAL_ANSWER_RT',
            'CURRENT_FIX_INTEREST_AREAS',
        ]
    ].copy()

    group_cols = ['participant_id', 'TRIAL_INDEX']
    rows = []

    for (participant_id, trial_index), g in df.groupby(group_cols):
        g = g.sort_values('CURRENT_FIX_END').copy()

        answer_rt = g['ANSWER_RT'].iloc[0]
        confirm_rt = g['CONFIRM_FINAL_ANSWER_RT'].iloc[0]

        first_fix_end = g['CURRENT_FIX_END'].iloc[0]
        first_fix_duration = g['CURRENT_FIX_DURATION'].iloc[0]
        first_fix_start = first_fix_end - first_fix_duration

        g['CURRENT_FIX_END_REL'] = g['CURRENT_FIX_END'] - first_fix_start
        g['ia_value'] = g['CURRENT_FIX_INTEREST_AREAS'].apply(_extract_single_ia_value)

        before_select = g[
            (g['CURRENT_FIX_END_REL'] <= answer_rt) &
            (g['ia_value'].notna())
        ]

        before_confirm = g[
            (g['CURRENT_FIX_END_REL'] <= confirm_rt) &
            (g['ia_value'].notna())
        ]

        last_ia_before_select = (
            before_select['ia_value'].iloc[-1]
            if not before_select.empty else pd.NA
        )

        last_ia_before_confirm = (
            before_confirm['ia_value'].iloc[-1]
            if not before_confirm.empty else pd.NA
        )

        rows.append({
            'participant_id': participant_id,
            'TRIAL_INDEX': trial_index,
            'ANSWER_RT': answer_rt,
            'CONFIRM_FINAL_ANSWER_RT': confirm_rt,
            'last_ia_before_select': last_ia_before_select,
            'last_ia_before_confirm': last_ia_before_confirm,
        })

    return pd.DataFrame(rows)


def attach_area_labels(
    extracted_df: pd.DataFrame,
    ia_df: pd.DataFrame,
) -> pd.DataFrame:
    ia_small = ia_df[
        ['participant_id', 'TRIAL_INDEX', 'IA_ID', 'area_label']
    ].copy()

    ia_small['IA_ID'] = pd.to_numeric(ia_small['IA_ID'], errors='coerce')

    extracted = extracted_df.copy()
    extracted['last_ia_before_select'] = pd.to_numeric(
        extracted['last_ia_before_select'], errors='coerce'
    )
    extracted['last_ia_before_confirm'] = pd.to_numeric(
        extracted['last_ia_before_confirm'], errors='coerce'
    )

    out = extracted.merge(
        ia_small.rename(columns={
            'IA_ID': 'last_ia_before_select',
            'area_label': 'area_label_before_select',
        }),
        on=['participant_id', 'TRIAL_INDEX', 'last_ia_before_select'],
        how='left',
    )

    out = out.merge(
        ia_small.rename(columns={
            'IA_ID': 'last_ia_before_confirm',
            'area_label': 'area_label_before_confirm',
        }),
        on=['participant_id', 'TRIAL_INDEX', 'last_ia_before_confirm'],
        how='left',
    )

    valid_pairs = ia_small[['participant_id', 'TRIAL_INDEX']].drop_duplicates()

    out = out.merge(
        valid_pairs,
        on=['participant_id', 'TRIAL_INDEX'],
        how='inner',
    )

    return out[
        [
            'TRIAL_INDEX',
            'participant_id',
            'area_label_before_select',
            'area_label_before_confirm',
        ]
    ]


def get_last_areas(hunt_path = '../../data/hunters.csv',
                   gath_path = '../../data/gatherers.csv',
                   fix_path = '../../data_raw/full/fixations_A.csv'):
    print("Loading hunters data...")
    hunt = load_raw_answers_data(hunt_path)
    print("Done!")
    print("Loading gatherers data...")
    gath = load_raw_answers_data(gath_path)
    print("Done!")

    print("Loading raw fixations data...")
    fix = load_raw_answers_data(fix_path)
    print("Done!")

    print("Searching for last IAs visited before selection and confirmation...")
    extracted = extract_last_ia_before_selection_and_confirmation(fix)
    print("Done!")

    print("Attaching hunters...")
    fin_hunt = attach_area_labels(extracted, hunt)

    print("Attaching gatherers...")
    fin_gath = attach_area_labels(extracted, gath)


    return fin_hunt, fin_gath


if __name__ == "__main__":
    hunters_last, gatherers_last = get_last_areas()

    hunters_last.to_csv("../../data/hunters_last.csv", index=False)
    gatherers_last.to_csv("../../data/gatherers_last.csv", index=False)

    print("Saved hunters_last and gatherers_last to ../../data")
