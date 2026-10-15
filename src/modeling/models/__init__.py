"""The model classes. Each has `fit`, `predict` and `get_coef_summary`.

    logreg_model.py   logistic regression on one row per trial -- the paper's model
    dummy_model.py    a baseline that ignores the features
    linreg_model.py   linear / ridge regression, for predicting answer reading
                      times (an exploration, not the paper)
    gbm_model.py      gradient-boosted trees, for the same exploration

The mixed-effects models (Julia and R) are in `explorations/mixed_models/`.
"""
