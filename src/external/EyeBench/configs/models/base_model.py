"""Minimal `BaseModelArgs` stub.

The external snippet imports this only for `replace_missing_values`, which is
not part of the trial-level feature extraction path. The two attributes it reads
are the per-row feature lists a downstream model would expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.external.EyeBench.configs.constants import (
    numerical_fixation_trial_columns,
    numerical_ia_trial_columns,
)


@dataclass
class BaseModelArgs:
    fixation_features: list[str] = field(
        default_factory=lambda: list(numerical_fixation_trial_columns)
    )
    ia_features: list[str] = field(
        default_factory=lambda: list(numerical_ia_trial_columns)
    )
