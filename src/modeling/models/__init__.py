"""The model classes. Each has `fit`, `predict` and `get_coef_summary`.

    logreg_model.py   logistic regression on one row per trial -- the paper's model
    glmm_model.py     the same model with random intercepts per participant and per item,
                      fitted in Julia; importing it starts Julia
    dummy_model.py    a baseline that ignores the features
    linreg_model.py   linear / ridge regression, for predicting answer reading
                      times (an exploration, not the paper)
    gbm_model.py      gradient-boosted trees, for the same exploration

The R mixed model (lme4) is parked in `explorations/mixed_models/`.
"""
