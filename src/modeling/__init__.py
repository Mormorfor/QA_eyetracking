"""Shared prediction machinery: folds, cross-validation, evaluation, inference, model backends.

Everything here is about *fitting and judging a model*, and nothing here knows
what an eye movement is. The split from `features/` is the one that matters:
`features/` decides what the columns are, `modeling/` decides what to do with
them. A feature definition appearing in this package is a bug.

Assembled in stage D step 7 (2026-10-07) out of `predictive_modeling/common/`
and the model-agnostic half of `predictive_modeling/answer_correctness/`, which
had grown into a folder named after one study's research question while holding
the machinery both studies and both strands use.

    folds.py        train/test splits, predefined fold assignments, the seven regimes
    crossval.py     the cross-validation loop and its persisted runs
    evaluate.py     metrics, fold aggregation, the prepared-dataset container
    inference.py    coefficient CIs (wald / bootstrap / clustered), VIF
    selection.py    feature-selection procedures
    feature_sets.py named feature-column sets -- what a model CONSUMES
    models/         the backends: logreg, dummy, julia, glmer_r, gbm, linreg
"""
