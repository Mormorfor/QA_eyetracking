"""The paper's headline model: predicting answer correctness from gaze.

The only analysis with subfolders. `person_variance/` is the per-participant
leave-one-trial-out work; `knowledge_regimes/` is Study 2. Diana, 2026-10-07:
*"Lets keep it all together for now, can separate later if needed"* -- so Study 2 is a
child here rather than a sibling analysis, which is also what
`lib/plotting/output.py::ANALYSES` already commits to.

The generic machinery this drives -- cross-validation, folds, metrics, the model
wrappers -- is `modeling/`, not here. What lives here is the choices.
"""
