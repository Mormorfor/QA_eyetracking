"""Predicting correctness without telling the model which answer is correct.

The paper's model is told which answer is correct: its features compare the correct answer
with the wrong ones. Here the same models are fitted on features that name the answers
without the answer key -- by screen position, by whether the participant selected them, or
summarised over all four -- so the model sees how the participant looked at the answers
but not which of them is right.

* `table.py` -- the trial table with the answers named those three ways, and the rule that
  turns a paper feature set into its versions on it.
* `run.py` -- each paper model and its three answer-free versions scored side by side, on
  Study 1 and Study 2, by the paper's own comparison; their confusion matrices, predicted
  probabilities and coefficients.
* `selected_features.py` -- deliberation features built around the selected answer, from
  the fixation sequence, the press times and the fixation report.
* `deliberation.py` -- groups of deliberation features added to the selected version, in
  every combination.
* `matching.py` -- whether the chosen answer stood out on each measure (the
  correctness-associations "matching" idea), as features added to the selected version.
* `tracker.py` -- the chosen set split into what needs no eye tracker, gaze times, and
  where the eyes went, to show what the eye tracker adds.
* `discrimination.py` -- whether right and wrong answers get different probabilities, and
  the model against the participants' own confidence rating.
* `showcase.py` -- the paper's model and the chosen answer-hidden set
  (`deliberation.CHOSEN_NAME`) side by side, in simple figures for slides.
* `cache.py` -- the slow results, saved once and read back while their inputs are the
  same. The answer-hidden tables are saved beside the paper's `model_ready.csv`.
* `overview.ipynb` -- walks through both.

Imports from `analyses/correctness_prediction/` so the scoring is the paper's own. Outputs go
to `reports/explorations/answer_hidden/`.
"""
