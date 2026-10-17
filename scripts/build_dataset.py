"""Build one dataset, from the raw reports to the model table.

    python scripts/build_dataset.py --dataset l1
    python scripts/build_dataset.py --dataset l1 --no-splits
    python scripts/build_dataset.py --dataset knowqa

Study 1 runs three steps, in this order:

  1. the answer screen: raw reports -> the three tables (`ingest/build.py`)
  2. the paragraph screen: -> `features/paragraph_spans.csv`
  3. the model table: -> `features/model_ready.csv`

Step 3 needs the other two, because it joins the paragraph features in. Study 2 goes through
`ingest/knowqa.py` instead, because its raw reports need converting and cleaning first, and it
has no paragraph screen. `docs/data-pipeline.md` describes both.

`--splits` (on by default, Study 1 only) also writes the hunters' and gatherers' subsets, of the
three tables and of the model table.

This file only calls functions defined elsewhere, in order. The two pilots are built by
`scripts/build_pilot.py` (see `main`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.datasets import DATASETS, dataset as get_dataset  # noqa: E402


#: Study 1's two groups. Each participant is in one group only, so the per-participant
#: features come out the same whether built per group or on everyone
#: (`docs/pitfalls.md` section 3). That is why no scope setting is needed below.
L1_FACETS = ("hunters", "gatherers")


def build_l1(verbose: bool = True, splits: bool = True) -> None:
    """Study 1: answer screen -> paragraph screen -> model table.

    `splits` says whether the hunters' and gatherers' subsets are written too:
    of the three tables (step 1) and of the model table (step 3).
    """
    from src.config import columns as C
    from src.features.build import load_model_input, save_model_ready
    from src.features.paragraph.spans import save_paragraph_features
    from src.ingest.build import main as ingest_main

    ds = get_dataset("l1")
    facets = L1_FACETS if splits else ()

    print(f"[1/3] ingest: raw reports -> 3 grain tables"
          f"{' + ' + '/'.join(facets) if facets else ' (no splits)'}")
    # `dataset=ds` makes it save the three tables, and `split_column` makes it
    # save each group's subsets of them too. Asking for the subsets does not
    # also write the single wide table (5.8 GB on Study 1).
    ingest_main(
        dataset=ds,
        split_column=C.QUESTION_PREVIEW_COLUMN if splits else None,
        verbose=verbose,
    )

    print("[2/3] features: paragraph spans -> the cache the trial table joins")
    save_paragraph_features(verbose=verbose)

    print("[3/3] features: trial-level model-ready table")
    # Reads the three tables step 1 wrote, joined back into one wide table.
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
    """Study 2: convert -> clean -> answer screen -> model table, in one call
    (`ingest/knowqa.py`). Study 2 has no paragraph screen, so there is no
    paragraph step.

    `splits` has no effect; it is accepted so both builders take the same
    arguments. Study 2 has no groups of people to split by: every participant
    does all three knowledge regimes, so splitting by regime would split each
    person's trials (`docs/pitfalls.md` section 3). `main()` prints a note
    saying so.
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
        # The two pilots are built by scripts/build_pilot.py, which uses the identity
        # scheme they were collected under. Building them here would use Study 2's
        # scheme instead and give a different dataset under the same file names, so
        # this refuses.
        raise SystemExit(
            f"{ds.key} is a {ds.kind}; build it with "
            f"`python scripts/build_pilot.py --pilot {ds.key}`."
        )

    if args.splits and args.dataset != "l1":
        print(f"note: --splits has no effect on {ds.key} -- no split column (see build_knowqa)")

    print(f"building {ds.label} ({ds.key}) -> {ds.root}")
    BUILDERS[args.dataset](verbose=not args.quiet, splits=args.splits)
    print(f"done: {ds.model_ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
