"""Can a trial's eye movements tell whether the participant had any knowledge?

Study 2 gives each participant three knowledge regimes before the answer screen: nothing,
the paragraph, or the correct answer's content in other words. Here a model is trained to
tell the two regimes with knowledge (partial and full) from the one without, labelled by
the regime alone, whether the answer was right or not.

* `run.py` -- the feature sets, leave-one-participant-out predictions, the scores and
  tests, and the figures.
* `guesses.py` -- right answers only: known or a lucky guess, and two ways of telling them
  apart better (training on right answers only, centring each feature within the person).
* `answer_free.py` -- partial and full knowledge each against none, with gaze features that
  do not know the answer key, set against response time.
* `overview.ipynb` -- walks through all three, showing every figure without saving it.

Imports Study 2's trial table and the paper's feature lists from
`analyses/correctness_prediction/` (`load_showcase_frame`, `SHOWCASE_FEATURE_COLS`,
`PERSON_LEVEL_COLS`), so the trials and features are the paper's. Outputs go to
`reports/explorations/knowledge_detection/`.
"""
