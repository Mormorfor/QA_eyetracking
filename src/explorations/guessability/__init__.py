"""How guessable is each question? Counted from people guessing the answers.

Prolific participants guessed the answers to the OneStop and RACE questions. An item
guessed right far more often than one in four gives its answer away without the text; one
guessed right far less often has a wrong answer that draws guessers to it. Not eye-tracked,
and not used in the paper.

* `data.py` -- the two datasets (`DATASETS`), one row per guess and one row per item, the
  histogram of items by how often they were guessed right, and the most and least guessed.
* `plots.py` -- the histogram.
* `overview.ipynb` -- both datasets, histogram and the most and least guessed items. Saves
  the figures and their tables.
* `pooled.py` -- the guessing files matched to the experiment's own questions, and every
  source's guesses pooled per question, KnowQA's no-knowledge answers included.
* `no_knowledge.py` -- the L1-trained model scored on KnowQA's no-knowledge trials with and
  without the most guessable questions. Imports the Study 2 analysis
  (`analyses/correctness_prediction/knowledge_regimes/`) so the model and figures are its own.
* `no_knowledge.ipynb` -- walks through both.

**Few guesses per item in `guess.tsv`.** About three, so an item can only score 0, 1/3,
2/3 or 3/3, and luck alone puts many items at both ends. Each histogram therefore carries
what it would look like if every item were equally guessable. The pilot has about sixteen
per item.

Outputs go to `reports/explorations/guessability/`.
"""
