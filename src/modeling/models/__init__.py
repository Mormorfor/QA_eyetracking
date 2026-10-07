"""Model backends, both strands in one place.

Merged in stage D step 7 from `answer_correctness/models/` and
`answer_RTs/models/`. They were two folders because two research strands wrote
them, not because a backend belongs to a strand -- `linreg_model.py`'s own header
says it mirrors `logreg_model.py`'s structure.

    logreg_model.py   trial-level logistic regression -- the paper's model
    dummy_model.py    baselines
    julia_model.py    mixed-effects via juliacall      [future directions]
    glmer_r_model.py  mixed-effects via R              [future directions]
    linreg_model.py   ridge regression, answer RTs     [parked strand]
    gbm_model.py      gradient boosting, answer RTs    [parked strand]
"""
