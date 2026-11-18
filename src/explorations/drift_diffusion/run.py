"""Every figure of this exploration, for one study or both.

    python -m src.explorations.drift_diffusion.run            # both studies
    python -m src.explorations.drift_diffusion.run knowqa     # Study 2 only

Run from the repo root. The first run on Study 1 reads its 1.8 GB fixation report, which takes
a few minutes; after that the placed fixations are read back from the cache (`data.py`).
"""

from __future__ import annotations

import sys
from typing import Optional

from src.explorations.drift_diffusion import ez, glam, signatures
from src.explorations.drift_diffusion.data import load_fixations, load_trials


def run(dataset_key: str, save: Optional[bool] = True) -> dict:
    """The model-free figures, the EZ fits and the GLAM fits for one study."""
    print(f"\n=== {dataset_key} ===")
    fix = load_fixations(dataset_key)
    trials = load_trials(dataset_key, fix)

    out = {"trials": trials}
    out["cascade"] = signatures.plot_gaze_cascade(fix, trials, dataset_key=dataset_key, save=save)
    out["cascade_by_condition"] = signatures.plot_cascade_by_condition(fix, trials, dataset_key=dataset_key, save=save)
    out["gaze_bias"] = signatures.plot_gaze_bias(trials, dataset_key=dataset_key, save=save)
    out["first_last"] = signatures.plot_first_last(trials, dataset_key=dataset_key, save=save)
    out["visits"] = signatures.plot_visit_durations(fix, trials, dataset_key=dataset_key, save=save)
    out["decision_times"] = signatures.plot_decision_times(trials, dataset_key=dataset_key, save=save)

    out["ez"] = ez.fit_ez(trials)
    out["ez_medians"] = ez.plot_ez(out["ez"], dataset_key=dataset_key, save=save)

    print("GLAM fits:")
    out["glam"] = glam.fit_all(trials, dataset_key)
    out["glam_check"] = glam.plot_glam_check(trials, out["glam"], dataset_key=dataset_key, save=save)
    out["glam_parameters"] = glam.plot_glam_parameters(out["glam"], dataset_key=dataset_key, save=save)
    return out


if __name__ == "__main__":
    for key in sys.argv[1:] or ["knowqa", "l1"]:
        run(key)
