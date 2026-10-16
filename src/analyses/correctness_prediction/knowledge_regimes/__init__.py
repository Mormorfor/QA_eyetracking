"""Study 2 (KnowQA): the L1-trained correctness model applied to the three knowledge
regimes, and its predicted probability set against the participants' own confidence.

* `descriptives` -- the data by regime (accuracy, pace, confidence, items, sittings), the
  L1 model fitted once and applied to every trial, and the figures built on its predictions.
* `comparison_runs` -- the L1 model scored on each regime as a separate run.
* `confidence_correlation` -- how the model's predicted probability relates to the
  confidence rating.

Six participants, three sessions each, 870 trials. Every participant does all three
regimes, so splitting by regime splits each person's trials (`pitfalls.md` §3).
"""
