"""Figures and tables for the headline model, one module per kind.

* `coefficients.py` -- coefficient bars, and a table comparing coefficients across
  models.
* `cv_comparison.py` -- cross-validation scores by regime.
* `feature_ranks.py` -- which features the per-participant models lean on, and how
  the features correlate.
* `model_comparison.py` -- several models' scores on every test of Study 1 or
  Study 2, one panel per test.
* `probabilities.py` -- the predicted probability of a correct answer, by outcome.
* `results.py` -- the scores of a fitted model, printed or as a table.
* `run_comparison.py` -- comparing saved runs, and comparing cross-validated models.

Every figure is saved by `lib/plotting/output.py::save_output`, with its numbers,
under `reports/correctness_prediction/`.
"""
