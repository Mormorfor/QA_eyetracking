"""Build one dataset, raw reports -> model-ready trial table.

    python scripts/build_dataset.py --dataset l1
    python scripts/build_dataset.py --dataset l1 --no-splits
    python scripts/build_dataset.py --dataset knowqa

`--splits` (on by default, L1 only) writes the hunters/gatherers subsets next to the
pooled tables, at both stages that can carry them -- the three grain tables and the
model-ready table. Until 2026-10-15 they were never written here at all.

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


#: L1's split column and the facet names it produces. `question_preview` is a
#: BETWEEN-participant manipulation, so each participant's whole trial set lands in
#: one facet and participant-level features come out the same either way
#: (`docs/pitfalls.md` section 3). That is why no scope argument is needed below.
L1_FACETS = ("hunters", "gatherers")


def build_l1(verbose: bool = True, splits: bool = True) -> None:
    """IA table -> paragraph cache -> model-ready. The three-stage L1 path.

    `splits` decides whether the hunters/gatherers subsets are written alongside
    the pooled tables, at BOTH stages that can carry them: the three grain tables
    in step 1 and the model-ready table in step 3. It used not to be a choice --
    the splits were simply never built here, because `split_column` was not passed,
    which is how `model_ready_{hunters,gatherers}.csv` came to be six days older
    than `model_ready.csv` on 2026-10-13.
    """
    from src.config import columns as C
    from src.features.build import load_model_input, save_model_ready
    from src.features.paragraph.spans import save_paragraph_features
    from src.ingest.build import main as ingest_main

    ds = get_dataset("l1")
    facets = L1_FACETS if splits else ()

    print(f"[1/3] ingest: raw reports -> 3 grain tables"
          f"{' + ' + '/'.join(facets) if facets else ' (no splits)'}")
    # `dataset=ds` is what makes the prep write the three grains (grain proposal
    # step 4d); `split_column` is what makes it write their facet subsets too.
    # The facet write is NOT gated on `write_broadcast_table`, so asking for splits
    # does not drag the 5.8 GB broadcast table back.
    ingest_main(
        dataset=ds,
        split_column=C.QUESTION_PREVIEW_COLUMN if splits else None,
        verbose=verbose,
    )

    print("[2/3] features: paragraph spans -> the cache the trial table joins")
    save_paragraph_features(verbose=verbose)

    print("[3/3] features: trial-level model-ready table")
    # Reads the three grain tables step 1 wrote, not the 2.9 GB broadcast table:
    # 4 s against 38 s, same frame (grain proposal step 6).
    save_model_ready(
        load_model_input(ds, verbose=verbose),
        output_path=ds.model_ready,
        dataset="l1",
        verbose=verbose,
    )
    for facet in facets:
        out = ds.model_ready_split(facet)
        print(f"      {facet} -> {out.name}")
        save_model_ready(
            load_model_input(ds, facet=facet, verbose=verbose),
            output_path=out,
            dataset="l1",
            verbose=verbose,
        )


def build_knowqa(verbose: bool = True, splits: bool = True) -> None:
    """Study 2's own path: convert -> clean -> pipeline -> features, in one call.

    `has_paragraph=False`, so there is no paragraph step to run -- that is a property of the
    dataset now rather than a flag a caller has to remember (stage D step 5).

    `splits` is accepted for a uniform signature and has no effect: KnowQA has no
    `question_preview` column to split on, and its three knowledge regimes are not
    the analogue -- they are WITHIN-participant, so slicing by them would split a
    participant's trials (`docs/pitfalls.md` section 3). Said out loud in `main()`
    rather than ignored silently.
    """
    from src.ingest.knowqa import prepare_know_qa

    prepare_know_qa(run="KnowQA", verbose=verbose)


BUILDERS = {"l1": build_l1, "knowqa": build_knowqa}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument(
        "--splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the hunters/gatherers subsets alongside the pooled tables "
             "(L1 only). --no-splits builds the pooled tables alone.",
    )
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

    if args.splits and args.dataset != "l1":
        print(f"note: --splits has no effect on {ds.key} -- no split column (see build_knowqa)")

    print(f"building {ds.label} ({ds.key}) -> {ds.root}")
    BUILDERS[args.dataset](verbose=not args.quiet, splits=args.splits)
    print(f"done: {ds.model_ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
