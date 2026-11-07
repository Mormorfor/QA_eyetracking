"""The correctness model as a mixed model: random intercepts per participant and per item.

    run.py                   what only the mixed model has: one fit on every trial with
                             its random effects, a fit report, and the variance summary
    plots.py                 the random-effect figures, and coefficients of several fits
                             side by side
    overview.ipynb           the mixed model through `../overview.ipynb`: cross-validated
                             accuracy against the logistic regression, the model
                             comparison, coefficients, random effects, one split
    knowledge_regimes.ipynb  the mixed model through the model half of
                             `../knowledge_regimes/overview.ipynb`: Study 1 fit applied
                             to Study 2, by regime, against confidence

The model is `modeling/models/glmm_model.py`. Everything it shares with the logistic
regression runs through the same functions, given `model_builder`. Needs Julia: run
`src/vendor/julia_integration/julia_test.ipynb` once first.
"""
