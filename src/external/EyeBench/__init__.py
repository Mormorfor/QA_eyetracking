"""Feature extraction code taken from the lab's EyeBench / OneStop pipeline.

Contents:

* `utils - paragraph feature extraction.py` -- the file as received from that
  project, kept as close to the original as possible so an updated copy can be
  dropped in. It imports its configuration from `src.configs.*`, which does not
  exist in this repo; the loader in `paragraph_trial_features.py` maps those
  imports onto `EyeBench/configs` instead (see `_alias_config_package`).
* `configs/` -- the constants that file imports, copied from the source project.
* `paragraph_trial_features.py` -- our adapter: bridges our raw paragraph
  reports to what the extraction expects, and is the module to import from.

Usage:
    from src.external.EyeBench.paragraph_trial_features import (
        save_paragraph_trial_level_features,
    )
"""
