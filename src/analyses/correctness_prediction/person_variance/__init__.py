"""How the correctness model differs from one participant to the next.

Here the model is fitted separately for each participant, on that person's trials
alone. Each of their trials is predicted by a model trained on their other trials --
one trial left out at a time, which the code calls "LOO" -- so every participant
gets an accuracy of their own. One more fit, on all of their trials, gives their
coefficients. This backs the paper's participant-level results;
`overview.ipynb`, beside this file, runs it and draws every figure.

    loo_runs.py                    runs the per-participant models, or reloads them
    person_accuracy.py             each participant's accuracy, and how it is spread
    coef_consistency.py            how the coefficients differ between participants
    univariate_consistency.py      the same question for every feature, with no model
    accuracy_characterization.py   what sets apart the people the model predicts well
    mistake_types.py               which kind of mistake the model makes, and on whom
    plot_style.py                  colours and labels shared by the figures
"""
