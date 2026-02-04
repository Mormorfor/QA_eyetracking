import pandas as pd
import numpy as np
import ast

import constants as C
from src.data_prep.data_csv_generation import scale_pupil_area_to_mm


def load_raw_answers_fix_data(ia_a_path = "data_raw/full/fixations_A.csv"):
    """
    Load raw fixation level answers data from CSV file.
    """
    return pd.read_csv(ia_a_path)


def compute_participant_pupil_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-participant mean and SD of pupil size (in mm).

    Steps:
    1. Clean pupil size column (replace '.' with NaN, cast to float)
    2. Scale pupil area to mm
    3. Compute mean and SD per participant
    """
    df_local = df.copy()

    # df_local[C.CURRENT_FIX_PUPIL_SIZE] = (
    #     df_local[C.CURRENT_FIX_PUPIL_SIZE]
    #     .replace(".", np.nan)
    #     .astype(float)
    # )

    df_local["pupil_mm"] = scale_pupil_area_to_mm(
        df_local[C.CURRENT_FIX_PUPIL_SIZE]
    )

    stats = (
        df_local
        .groupby(C.PARTICIPANT_ID)["pupil_mm"]
        .agg(
            pupil_mean="mean",
            pupil_sd="std"
        )
        .reset_index()
    )

    return stats



def main():

    input_path = "data_raw/full/fixations_A.csv"
    output_path = "data/participant_pupils.csv"

    df = load_raw_answers_fix_data(input_path)
    participant_pupil_stats = compute_participant_pupil_stats(df)
    participant_pupil_stats.to_csv(output_path, index=False)

    print(
        f"Saved participant-level pupil statistics "
        f"({len(participant_pupil_stats)} participants) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
