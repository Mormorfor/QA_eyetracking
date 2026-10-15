"""Fitting models and judging them: folds, cross-validation, scores, confidence intervals.

Nothing here knows what an eye movement is. `features/` decides what the
columns are; this folder decides what to do with them. A feature definition
does not belong here.

    feature_sets.py  named lists of the columns a model uses
    folds.py         which trials train and which are held out
    crossval.py      the cross-validation loop, and saving its results
    evaluate.py      fit-and-score on one split, and the summary tables
    inference.py     confidence intervals for coefficients, and VIF
    selection.py     three ways to pick features
    models/          the model classes
"""
