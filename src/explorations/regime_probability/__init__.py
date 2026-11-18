"""Does the paper's model give correct answers a lower probability when they were guessed?

On Study 2, the correctness model trained on Study 1 is applied to every trial. A correct
answer under no knowledge is a guess, so a model that reads behaviour rather than outcome
should give it a lower probability than a correct answer given with knowledge. This tests
that: the mean predicted probability on correct answers in each knowledge regime, and the
differences between regimes.

* `run.py` -- the predictions, the means, the tests and the figure.
* `overview.ipynb` -- walks through it, showing the figure without saving it.

Imports the paper's Study 2 path from `analyses/correctness_prediction/knowledge_regimes/`
(loading the table, fitting on Study 1, predicting), so the model tested is exactly the
paper's. Outputs go to `reports/explorations/regime_probability/`.
"""
