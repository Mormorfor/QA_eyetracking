"""Build one of Study 2's two pilots, from the raw reports to the model table.

    python scripts/build_pilot.py --pilot second_test
    python scripts/build_pilot.py --pilot testrun_qa

A minimal runner, kept so the pilots stay buildable. Nothing in the paper uses them.

Three steps, each pilot on its own:

  1. convert: the raw DataViewer exports -> plain CSVs in `<raw>/csvs/`
  2. clean: the columns renamed to the names the pipeline expects -> `<raw>/csvs/cleaned/`
  3. build: the cleaned reports -> the three tables (`ingest/build.py`), then the model table
     (`features/build.py`), all under the pilot's folder in `data/datasets/pilots/`

The pilots are not built by `ingest/knowqa.py` because they use an older identity scheme: the
recording label is the `participant_id` as it is, and `TRIAL_INDEX` is the tracker's own trial
counter. `knowqa.py` splits the label into person and session, and would give a different
dataset under the same file names. The conversion and the column renames are shared with it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.config import columns as Con  # noqa: E402
from src.config.datasets import Dataset, dataset, pilots  # noqa: E402


def clean_report(df: pd.DataFrame) -> pd.DataFrame:
    """One converted pilot report, with the column names and values the pipeline expects.

    The same renames as KnowQA, plus the recording label as `participant_id` (lower-cased,
    as every join downstream expects); `answers_order` as letters; and `same_critical_span`
    from the end of `text_id_with_q`, as `knowqa.add_item_id_columns` explains.
    """
    from src.ingest import knowqa

    df = df.rename(columns={**knowqa.RENAME_MAP, Con.RECORDING_SESSION_LABEL: Con.PARTICIPANT_ID})
    df[Con.ANSWERS_ORDER_COLUMN] = df[Con.ANSWERS_ORDER_COLUMN].apply(knowqa.answers_order_to_letters)
    df[Con.PARTICIPANT_ID] = df[Con.PARTICIPANT_ID].astype(str).str.strip().str.lower()
    return knowqa.add_item_id_columns(df, verbose=False)


def build_pilot(ds: Dataset, verbose: bool = True) -> Path:
    """Convert, clean and build pilot `ds`. Returns the path of its model table."""
    from src.features.build import load_model_input, save_model_ready
    from src.ingest import knowqa
    from src.ingest import registry
    from src.ingest.build import main as ingest_main

    print(f"[1/3] convert: {ds.raw_dir}")
    knowqa.convert_raw_reports(ds.raw_dir, verbose=verbose)

    print("[2/3] clean")
    csv_dir = ds.raw_dir / knowqa.CSV_SUBDIR
    cleaned_dir = csv_dir / knowqa.CLEANED_SUBDIR
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    for name in knowqa.CLEANED_REPORT_NAMES:
        clean_report(pd.read_csv(csv_dir / name, low_memory=False)).to_csv(
            cleaned_dir / name, index=False
        )

    print(f"[3/3] build -> {ds.root}")
    ds.aux.mkdir(parents=True, exist_ok=True)
    ingest_main(
        dataset=ds,
        ia_answers_path=cleaned_dir / knowqa.IA_REPORT_NAME,
        output_path=ds.all_participants,
        # One fixation report feeds the pupil baseline, the fixation sequence, the
        # reading times and the button presses (it holds the message log too).
        fixations_path=cleaned_dir / knowqa.FIX_REPORT_NAME,
        pupil_stats_path=ds.pupil_baseline_answers,
        button_clicks_path=ds.button_clicks,
        button_clicks_fix_tsv_path=None,
        button_clicks_msg_participant_col=Con.PARTICIPANT_ID,
        all_answers_is_cumulative=False,  # each trial's ALL_ANSWERS is its own
        last_labels_path=ds.aux / "all_participants_last.csv",
        rt_and_tfd_path=ds.aux / "RT_and_TFD.csv",
        base_function_names=[
            name for name, entry in registry.FUNCTION_REGISTRY.items()
            if entry["kind"] == "row" and name not in ds.skip_base_features
        ],
        # No repeated-reading trials (and no column for them); practice trials are removed.
        remove_repeats=False,
        verbose=verbose,
    )
    save_model_ready(
        load_model_input(ds, verbose=verbose),
        output_path=ds.model_ready,
        dataset=ds,
        verbose=verbose,
    )
    return ds.model_ready


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", required=True, choices=[d.key for d in pilots()])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    build_pilot(dataset(args.pilot), verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
