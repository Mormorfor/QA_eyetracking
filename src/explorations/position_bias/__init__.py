"""Does where an answer sits on screen, or when the opening scan reaches it, pull the choice
towards it?

The question arose from the opening scan: nearly everyone reads the question and then looks at
the answer right below it first, the top one, and usually goes round from there. If where the
eyes go shapes the choice, the answer met first should be chosen more often whenever there is
nothing better to go on. Study 2's no-knowledge regime is that case; the regimes with knowledge
and Study 1 are the comparisons.

* `data.py` -- one row per trial and answer, for either study: where it sat, its kind, its
  length, when the scan first reached it, its share of the gaze, how often other people chose
  it, and whether it was chosen.
* `placement.py` -- the check that nothing was put at the top, or anywhere, more often than
  chance: not the correct answer, not any kind of answer, not long answers, not answers others
  chose.
* `choice.py` -- the tests: where the chosen answer sat, where the wrong answers went, whether
  accuracy depends on where the correct answer sat, and conditional logits for the sizes.
* `scan.py` -- what the eyes do with each position: first looks, each person's usual first look,
  gaze on the answers not chosen, visits, returns from the question; and speed and confidence.
* `plots.py` -- the figures.
* `run.py` -- everything, for one study or both, saved.
* `overview.ipynb` -- walks through it all, figure by figure. Shows without saving.

**Why the tests need no model.** Both studies place the answers at random, so the answer at any
one position is a random one. If position made no difference, the chosen answer would sit at
each position one time in four, exactly, whatever the answers are like and however people
differ. Each test re-draws the placement the way the design drew it -- per trial in Study 2,
per item and list in Study 1 (`data.UNIT_COLS`) -- and asks how often chance does as well as the
data.

Uses the answer-screen fixations and decision times of `explorations/drift_diffusion/data.py`.
Outputs go to `reports/explorations/position_bias/`.
"""
