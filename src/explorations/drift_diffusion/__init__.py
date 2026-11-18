"""Drift-diffusion models of the answer choice, from the simplest up to ones that use the eyes.

A drift-diffusion model treats a choice as evidence building up, noisily, until it reaches a
threshold; the attentional versions (the aDDM of Krajbich et al. 2010 and Krajbich & Rangel
2011) let an option build up evidence faster while it is looked at. This folder applies that
family to the answer screen, in order of how much the model assumes:

* `signatures.py` -- no model: the gaze-and-choice patterns those papers start from (the
  gaze cascade, choice against relative gaze, the first and last answer looked at, visit
  durations, decision times).
* `ez.py` -- EZ-diffusion: a two-outcome (right / wrong) drift-diffusion model per
  participant, in closed form, without the eyes.
* `glam.py` -- GLAM: a race between the four answers in which gaze scales each answer's
  evidence, fitted exactly per group or knowledge regime, with and without the gaze term.
* `data.py` -- the answer-screen fixations and the per-trial decision time, choice and gaze
  shares everything above uses.
* `run.py` -- every figure, for one study or both, saved.
* `overview.ipynb` -- walks through all of it, explaining each figure: what is plotted, how to
  read it, and what the models predict. Shows without saving.

The decision is the final selection press, and only gaze before it enters a model (`data.py`
says why). Outputs go to `reports/explorations/drift_diffusion/`.
"""
