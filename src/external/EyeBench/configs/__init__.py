"""Configuration for the vendored EyeBench feature extraction.

Mirrors the `src.configs` package of the source project, reproducing only the
names `utils - paragraph feature extraction.py` imports. That file's
`from src.configs...` imports are redirected here at load time by
`src/external/EyeBench/paragraph_trial_features.py`, so the vendored file itself
needs no editing. Project-wide constants live in `src/constants.py`.
"""
