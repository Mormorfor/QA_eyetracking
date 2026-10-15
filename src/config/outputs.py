"""Where results and caches are written.

Figures and tables are not listed here: they are saved by
`lib/plotting/output.py::save_output`, which works out their folder from the
analysis name. This file holds the few other locations.
"""

from src.config.datasets import PROJECT_ROOT

REPORTS_ROOT = PROJECT_ROOT / "reports"

# Hand-written lists of which features each model uses. The code reads these and
# never writes them -- they are decisions, not results. See configs/README.md.
FEATURE_SETS_DIR = PROJECT_ROOT / "configs" / "feature_sets"

# Saved results that take a long time to recompute. Not meant for reading.
CACHE_ROOT = PROJECT_ROOT / "reports" / "_cache"
CROSS_VALIDATION_RUNS_DIR = CACHE_ROOT / "cross_validation_runs"
PER_PERSON_LOO_RESULTS_DIR = CACHE_ROOT / "per_person_corr_loo_results"
