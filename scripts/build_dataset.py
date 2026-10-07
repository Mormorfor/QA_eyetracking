"""Build one dataset, raw reports -> model-ready trial table.

    python scripts/build_dataset.py --dataset l1
    python scripts/build_dataset.py --dataset knowqa

This is where the build order stops being documentation and becomes code (`todo.md` T5.6).
`docs/data-pipeline.md` section 6 describes it; until now the only thing enforcing it was a
`FileNotFoundError` telling you to build the paragraph cache first.

**The order, and why it is this order.** The paragraph cache has to exist before the
model-ready table is assembled, because `build_trial_level_model_df` joins paragraph dwell
proportions in. Both studies share the trial-table step; they differ only in how the raw
reports become an IA table, which is why L1 goes through `ingest.build` and KnowQA through
`ingest.knowqa` (composite trial ids, per-session cleaning, its own identity scheme).

Nothing here is new logic -- every step is a call to a function that already existed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.datasets import DATASETS, dataset as get_dataset  # noqa: E402


def build_l1(verbose: bool = True) -> None:
    """IA table -> paragraph cache -> model-ready. The three-stage L1 path."""
    import pandas as pd

    from src.features.build import save_model_ready
    from src.features.paragraph.spans import save_paragraph_features
    from src.ingest.build import main as ingest_main

    ds = get_dataset("l1")

    print("[1/3] ingest: raw reports -> all_participants.csv (+ hunters/gatherers)")
    ingest_main(verbose=verbose)

    print("[2/3] features: paragraph spans -> the cache the trial table joins")
    save_paragraph_features(verbose=verbose)

    print("[3/3] features: trial-level model-ready table")
    ia = pd.read_csv(ds.all_participants, dtype={"participant_id": str}, low_memory=False)
    save_model_ready(ia, output_path=ds.model_ready, dataset="l1", verbose=verbose)


def build_knowqa(verbose: bool = True) -> None:
    """Study 2's own path: convert -> clean -> pipeline -> features, in one call.

    `has_paragraph=False`, so there is no paragraph step to run -- that is a property of the
    dataset now rather than a flag a caller has to remember (stage D step 5).
    """
    from src.ingest.knowqa import prepare_know_qa

    prepare_know_qa(run="KnowQA", verbose=verbose)


BUILDERS = {"l1": build_l1, "knowqa": build_knowqa}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ds = get_dataset(args.dataset)
    if args.dataset not in BUILDERS:
        # The two pilots are built by notebooks/builders/data_prep_new_exp.ipynb, which
        # carries the older identity scheme they were collected under. Driving them from
        # here would rebuild them under KnowQA's scheme instead -- a different dataset with
        # the same filename. Refuse rather than guess (`todo.md` T5.8, T5.10).
        raise SystemExit(
            f"{ds.key} is a {ds.kind}; it is built by "
            f"notebooks/builders/data_prep_new_exp.ipynb, not by this script. "
            f"See docs/decisions/2026-10-07-stage-f-proposal.md."
        )

    print(f"building {ds.label} ({ds.key}) -> {ds.root}")
    BUILDERS[args.dataset](verbose=not args.quiet)
    print(f"done: {ds.model_ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
