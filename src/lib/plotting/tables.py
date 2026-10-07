"""The `tables=` payload shared by figures that carry a summary and a test.

Moved out of `viz/viz_helpers.py` in stage E step 2. It is in `lib/` because it
names no project vocabulary -- only the two output-table keys, `summary` and
`fisher` -- and because two different analyses use it: the correctness-association
plots and `correctness_prediction/knowledge_regimes`.

It replaced `save_plot_and_report`, which was a second save path alongside
`save_output` (`todo.md` T1.5). Saving belongs to `save_output`; this only decides
*what* travels with the figure.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def correctness_tables(
    summary_df: pd.DataFrame, test_res: Optional[Dict]
) -> Dict[str, object]:
    """Assemble the ``tables=`` payload: the summary, plus the test if one ran."""
    tables: Dict[str, object] = {"summary": summary_df}
    if test_res is not None:
        tables["fisher"] = dict(test_res)
    return tables
