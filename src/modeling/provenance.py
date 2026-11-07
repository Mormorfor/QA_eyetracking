"""Which model made a set of predictions, so that everything saved from them says so.

Every model class has a `family`: "logreg" for the logistic regression, "glmm" for the
mixed model. Whatever makes predictions writes it into its tables as a `model_family`
column (cross-validation, the model comparison, the Study 2 predictions, the error
analysis). Whatever saves figures or tables from those predictions puts them in a folder
named after it, `reports/<analysis>/{figures,tables}/<family>/...`, through
`model_folder`.

So two models' outputs never share a file, and a figure cannot be saved under one model
while showing another's numbers. A table with no `model_family` stops the save rather
than being filed under a guess.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

MODEL_FAMILY_COL = "model_family"


def model_folder(data: Union[pd.DataFrame, pd.Series], subdir: Optional[str] = None) -> str:
    """The folder for outputs made from `data`: its model family, then `subdir` if given.

    `data` is a table with a `model_family` column, or a small summary (a Series, such
    as quadrant counts) carrying the family in `data.attrs["model_family"]`. A table that
    mixes models, such as a side-by-side comparison, goes in a folder naming all of them,
    joined by "+" ("glmm+logreg").
    """
    if isinstance(data, pd.DataFrame) and MODEL_FAMILY_COL in data.columns:
        families = sorted(str(f) for f in pd.unique(data[MODEL_FAMILY_COL].dropna()))
        if not families or data[MODEL_FAMILY_COL].isna().any():
            raise ValueError(f"Rows with no {MODEL_FAMILY_COL!r}: cannot tell which model made them.")
    elif MODEL_FAMILY_COL in data.attrs:
        families = [str(data.attrs[MODEL_FAMILY_COL])]
    else:
        raise KeyError(
            f"This table does not say which model made it (no {MODEL_FAMILY_COL!r}), so "
            f"there is no folder to save it in. Tables made by the project's prediction "
            f"functions carry it; a table built by hand needs it added."
        )
    return "/".join(x for x in ("+".join(families), subdir) if x)
