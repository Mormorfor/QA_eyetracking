import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import LogisticRegression


def extract_logreg_coefficients(
    model: LogisticRegression,
    feature_names: list[str],
    standardize: bool = False,
    X_train: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Extract coefficients from a fitted LogisticRegression model.

    If standardize=True, X_train must be provided and coefficients
    are scaled by feature standard deviation.
    """
    if model.coef_.shape[0] != 1:
        raise ValueError("Only binary LogisticRegression is supported.")

    coef = model.coef_[0]

    if standardize:
        if X_train is None:
            raise ValueError("X_train required for standardized coefficients.")
        scale = X_train.std(axis=0).replace(0, np.nan).to_numpy()
        coef_std = coef * scale
    else:
        coef_std = None

    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "odds_ratio": np.exp(coef),
        "abs_coef": np.abs(coef),
    })

    if coef_std is not None:
        df["standardized_coef"] = coef_std
        df["abs_standardized_coef"] = np.abs(coef_std)

    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
