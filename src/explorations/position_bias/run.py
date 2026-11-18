"""Every table and figure of this exploration, for one study or both, saved.

    python -m src.explorations.position_bias.run            # both studies, then the overview
    python -m src.explorations.position_bias.run knowqa     # Study 2 only

Run from the repo root. Study 1 takes several minutes, most of it resampling its 360
participants for the model intervals.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Sequence

import pandas as pd

from src.explorations.drift_diffusion.data import load_fixations, load_trials
from src.explorations.position_bias import choice, placement, plots, scan
from src.explorations.position_bias.data import answers_table

N_DRAWS = 10000  # random re-placements per test
N_BOOT = {"knowqa": 2000, "l1": 1000}  # participant resamples per model; Study 1's fits are slower


def compute(dataset_key: str, *, n_draws: int = N_DRAWS, n_boot: Optional[int] = None, seed: int = 0,
            verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Every table for one study. The per-answer table comes back too, as `answers`."""
    fix = load_fixations(dataset_key, verbose=verbose)
    trials = load_trials(dataset_key, fix, verbose=verbose)
    answers = answers_table(dataset_key, fix, trials)
    usual, usual_table = scan.usual_starts(answers, dataset_key)
    speed, speed_tests = scan.speed_table(answers, dataset_key)
    t = {
        "answers": answers,
        "placement_tests": placement.placement_tests(answers, dataset_key, n_draws=n_draws, seed=seed),
        "placement_kinds": placement.kind_table(answers, dataset_key),
        "placement_length_attractiveness": placement.length_attractiveness_table(answers, dataset_key),
        "placement_expected": placement.expected_shares(answers, dataset_key),
        "choice_tests": choice.randomisation_tests(answers, dataset_key, usual, n_draws=n_draws, seed=seed),
        "choice_positions": choice.position_table(answers, dataset_key),
        "choice_accuracy": choice.accuracy_table(answers, dataset_key),
        "choice_order": choice.order_table(answers, dataset_key),
        "choice_by_participant": choice.participant_table(answers, dataset_key),
        "choice_kinds": choice.kind_table(answers, dataset_key),
        "choice_length": choice.length_table(answers, dataset_key),
        "top_summary": choice.top_summary(answers, dataset_key),
        "models": choice.choice_models(answers, dataset_key, n_boot=n_boot or N_BOOT[dataset_key], seed=seed),
        "usual_starts": usual_table,
        "usual_start_choices": scan.choices_by_start(answers, dataset_key, usual),
        "first_looks": scan.first_look_table(answers, dataset_key),
        "gaze_by_position": scan.gaze_table(answers, dataset_key, "position"),
        "gaze_by_order": scan.gaze_table(answers, dataset_key, "order"),
        "returns": scan.returns_table(fix, answers, dataset_key),
        "speed": speed,
        "speed_tests": speed_tests,
        "deviation": scan.deviation_table(answers, dataset_key),
    }
    if dataset_key == "knowqa":
        t["confidence"], t["confidence_tests"] = scan.confidence_table(answers, dataset_key)
    if verbose:
        no_gaze = int((answers["excluded"] != "").sum() // 4)
        print(f"{dataset_key}: {len(answers) // 4} trials, {no_gaze} without gaze before a decision "
              "(in the choice tests, not the gaze analyses)")
        show = ["subset", "test", "n_trials", "value", "p"]
        print(t["choice_tests"][show].round(4).to_string(index=False))
    return t


def figures(t: Dict[str, pd.DataFrame], dataset_key: str, *, save: Optional[bool] = True) -> None:
    """Every figure for one study."""
    plots.plot_placement(t, dataset_key, save=save, close=True)
    plots.plot_choices(t, dataset_key, save=save, close=True)
    if dataset_key == "knowqa":
        plots.plot_guesses(t, save=save, close=True)
    plots.plot_models(t, dataset_key, save=save, close=True)
    plots.plot_scan(t, dataset_key, save=save, close=True)
    plots.plot_side(t, dataset_key, save=save, close=True)


def run(dataset_keys: Sequence[str] = ("knowqa", "l1"), *, save: Optional[bool] = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Every table and figure for each study, and the overview when both are run."""
    out = {}
    for key in dataset_keys:
        print(f"\n=== {key} ===")
        out[key] = compute(key)
        figures(out[key], key, save=save)
    if set(dataset_keys) == {"knowqa", "l1"}:
        plots.plot_overview(out, save=save, close=True)
    return out


if __name__ == "__main__":
    run(sys.argv[1:] or ("knowqa", "l1"))
