"""The paper's headline model: predicting from eye movements whether a participant
picks the correct answer.

* `run.py` -- fit the logistic regression on one train/test split, score it, and
  draw its figures.
* `comparison.py` -- any set of models scored side by side: Study 1 by
  cross-validation, Study 2 by one fit on all of Study 1.
* `plots/` -- those figures, the cross-validation figures, and the comparisons of
  runs and models.
* `report.py` -- collect saved runs and compare them.
* `participant_level.py` -- the same model fitted on one participant at a time.
* `person_variance/` -- what those per-participant models show.
* `knowledge_regimes/` -- Study 2 (KnowQA).

The general machinery -- folds, cross-validation, the model, coefficient intervals --
is in `modeling/`. What is here is how this analysis uses it. Outputs go to
`reports/correctness_prediction/`.
"""
