"""Where generated artefacts are written.

Separate from `datasets.py` on purpose: these are *destinations*, and keeping
them apart is what stops a report path being hardcoded at a call site.

Figures and tables do not appear here -- they go through
`viz/plot_output.py::save_output`, which derives their location from the
analysis name. What is left is the handful of caches that are not figures.
"""

from src.config.datasets import PROJECT_ROOT

REPORTS_ROOT = PROJECT_ROOT / "reports"

# Hand-written configuration the code READS, never writes. Not a result -- see
# configs/README.md. Was reports/report_data/answer_correctness/feature_columns
# until 2026-10-06, which framed the paper's five feature sets as output of the
# feature search when they are hand-specified inputs to it.
FEATURE_SETS_DIR = PROJECT_ROOT / "configs" / "feature_sets"

# Caches, not reports: big enough that regenerating is expensive, not
# interesting enough to read. Both pointed into the deleted report_data/ tree
# until 2026-10-06 and resolved to nothing.
CACHE_ROOT = PROJECT_ROOT / "reports" / "_cache"
CROSS_VALIDATION_RUNS_DIR = CACHE_ROOT / "cross_validation_runs"
PER_PERSON_LOO_RESULTS_DIR = CACHE_ROOT / "per_person_corr_loo_results"
