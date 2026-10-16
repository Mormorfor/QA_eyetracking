"""Run one analysis end to end: figures and the numbers behind them.

    python scripts/run_analysis.py --list
    python scripts/run_analysis.py --name scan_strategies
    python scripts/run_analysis.py --name attention_allocation --no-save

`--name` is validated against `lib/plotting/output.py::ANALYSES`, the same registry
`save_output` validates against -- so a typo fails here the way it would fail there, rather
than inventing a folder.

**What this is not.** It is not a replacement for the driver notebooks: several analyses have
parameters worth varying interactively, and this runs them at their defaults. It is the
answer to "can a reader regenerate this without opening Jupyter", which the release standard
in `conventions.md` requires and which was not true before stage F.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# analysis -> [(module, function), ...]. Every runner listed here takes the IA-level frame
# as its first argument; the one that does not (`run_all_area_mixed_models`, which wants the
# two group frames) is handled in `_run` below.
RUNNERS: dict[str, list[tuple[str, str]]] = {
    "attention_allocation": [
        ("src.analyses.attention_allocation.plots", "run_all_area_barplots"),
        ("src.analyses.attention_allocation.plots", "run_all_area_metric_plots"),
        ("src.analyses.attention_allocation.stats", "run_all_area_mixed_models"),
    ],
    "scan_strategies": [
        ("src.analyses.scan_strategies.plots", "run_all_strategy_plots"),
        ("src.analyses.scan_strategies.plots", "run_all_simplified_visit_matrices"),
        ("src.analyses.scan_strategies.dominant_eye", "run_dominant_strategy_eye_analysis"),
    ],
    "correctness_associations": [
        ("src.analyses.correctness_associations.plots", "run_all_correctness_seq_len_threshold_plots"),
        ("src.analyses.correctness_associations.plots", "run_all_correctness_seq_len_continuous_plots"),
        ("src.analyses.correctness_associations.plots", "run_all_trial_mean_dwell_threshold_plots"),
        ("src.analyses.correctness_associations.plots", "run_all_trial_mean_dwell_continuous_plots"),
        ("src.analyses.correctness_associations.plots", "run_all_total_answering_rt_continuous_plots"),
        ("src.analyses.correctness_associations.plots", "run_all_back_and_forth_pattern_plots"),
        ("src.analyses.correctness_associations.plots", "run_all_matching_correctness_plots"),
    ],
    "last_visitation": [
        ("src.analyses.last_visitation.plots", "run_all_last_label_before_confirm_plots"),
    ],
}

# Analyses that exist but have no single-call runner, with the reason. Listed rather than
# omitted so `--list` tells the whole truth.
NO_RUNNER: dict[str, str] = {
    "answer_rt_comparison": "one plotting function, called directly; takes the trial table",
    "text_qa_relationship": "a sequence of map/contrast calls; driven by notebooks/drivers/text_associations.ipynb",
    "correctness_prediction": "cross-validation and per-person runs; expensive, driven by its notebooks",
    "correctness_prediction/person_variance": "needs the cached LOO pickle; see notebooks/drivers/per_person_runs.ipynb",
    "correctness_prediction/knowledge_regimes": "Study 2; see notebooks/drivers/preliminary_analysis.ipynb",
}


def _run(name: str, save: bool, verbose: bool) -> None:
    import importlib

    import pandas as pd

    from src.config.datasets import dataset as get_dataset

    from src.features.build import load_model_input

    ds = get_dataset("l1")

    # Reads the three grain tables and rejoins them, rather than the 2.9 GB
    # broadcast table: 4 s against 64 s, and the frames are **verified equivalent**
    # on every column the analyses can see -- 121 shared columns, 0 value, 0 null
    # and 0 dtype differences (2026-10-13). The 118 columns the read-time whitelist
    # drops are raw passthrough that nothing under `analyses/` reads.
    if verbose:
        print("loading the grain tables")
    all_participants = load_model_input(ds, verbose=verbose)

    for mod_name, fn_name in RUNNERS[name]:
        fn = getattr(importlib.import_module(mod_name), fn_name)
        if verbose:
            print(f"  {fn_name}")
        if fn_name == "run_all_area_mixed_models":
            # The one runner that wants the two group frames rather than the pooled
            # one. It read `ds.root / "hunters.csv"` until 2026-10-13 -- a path that
            # has not existed since stage G moved those files under `interim/`, so
            # this branch raised FileNotFoundError. The group subsets of each grain
            # are written in the same run as their parent, so they cannot be stale
            # relative to it the way the old pair could.
            fn(
                load_model_input(ds, facet="hunters", verbose=False),
                load_model_input(ds, facet="gatherers", verbose=False),
                save=save,
            )
        else:
            fn(all_participants, save=save)


def main(argv: list[str] | None = None) -> int:
    from src.lib.plotting.output import ANALYSES

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list or not args.name:
        print("runnable from here:")
        for k in sorted(RUNNERS):
            print(f"  {k:42s} {len(RUNNERS[k])} step(s)")
        print("\nregistered but not runnable from here:")
        for k, why in sorted(NO_RUNNER.items()):
            print(f"  {k:42s} {why}")
        unknown = sorted(set(ANALYSES) - set(RUNNERS) - set(NO_RUNNER))
        if unknown:
            print("\nin ANALYSES but unaccounted for here:")
            for k in unknown:
                print(f"  {k}")
        return 0

    if args.name not in ANALYSES:
        raise SystemExit(f"{args.name!r} is not in ANALYSES; see --list.")
    if args.name not in RUNNERS:
        raise SystemExit(f"{args.name!r}: {NO_RUNNER.get(args.name, 'no runner registered')}")

    _run(args.name, save=not args.no_save, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
