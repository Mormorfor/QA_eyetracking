"""Bundle a summary table and a test result for `save_output(tables=...)`.

Used by figures that show a summary and a significance test together. This only
decides what goes into `tables=`; the saving itself is done by `save_output`.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def correctness_tables(
    summary_df: pd.DataFrame, test_res: Optional[Dict]
) -> Dict[str, object]:
    """``{"summary": ..., "fisher": ...}``: the summary, plus the test result if one ran."""
    tables: Dict[str, object] = {"summary": summary_df}
    if test_res is not None:
        tables["fisher"] = dict(test_res)
    return tables
