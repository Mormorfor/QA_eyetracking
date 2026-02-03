import pandas as pd
import numpy as np
import ast

import constants as C


def load_raw_answers_fix_data(ia_a_path = "data_raw/full/fixations_A.csv"):
    """
    Load raw fixation level answers data from CSV file.
    """
    return pd.read_csv(ia_a_path)


def compute_participant_pupil_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-participant mean and standard deviation of pupil size.

    Returns a DataFrame with:
        PARTICIPANT_ID | pupil_mean | pupil_sd
    """
    stats = (
        df
        .groupby(C.PARTICIPANT_ID)[C.CURRENT_FIX_PUPIL_SIZE]
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
