# notebooks/experiment

How the Study 2 materials were made. These notebooks are kept as the record of what was
built, not to be run again.

- `extract_text.ipynb`: the texts. One row per Study 1 item (paragraph, question, four
  answers), with the hand-written rephrased correct answers added:
  `experiment_text_completed.csv`.
- `text_adjustments.ipynb`: copies the rephrased correct answers from the advanced texts to
  the simplified ones, and checks their spelling (`src/experiment/texts.py`).
- `lists_builder.ipynb`: the 54 ordered trial lists (`src/experiment/lists.py` explains the
  design), saved per batch to `data/stimuli/new_batched_lists/`, with a short version for
  trying the experiment out in `SMALL_DEMO/`.

The functions are also in `src/experiment/`, where a new experiment can start from them; the
notebooks keep their own copies. The materials are in `data/stimuli/`, and only
`lists_builder` looks for them there: `extract_text` reads and writes in `data/` itself (its
Study 1 table is in `data/datasets/l1_onestop/interim/`), and `text_adjustments` in
`../data/Experiment`, which does not exist.

The lists come out the same again only from the same input rows in the same order; see
`src/experiment/lists.py`.
