import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.data_paths import FIX_ANSWERS_PATH, PARTICIPANT_PUPILS_PATH
from src.derived.pupil_norm import compute_participant_pupil_stats


def load_raw_answers_fix_data(ia_a_path: Path = FIX_ANSWERS_PATH):
    """
    Load raw fixation level answers data from CSV file.
    """
    return pd.read_csv(ia_a_path)


def main(
    input_path: Path = FIX_ANSWERS_PATH,
    output_path: Path = PARTICIPANT_PUPILS_PATH,
):
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
