"""Saving the slow results of this exploration once, and reading them back after that.

`cached(name, inputs, compute)` runs `compute()` the first time and saves what it returns,
under a name made from `name` and a fingerprint of `inputs`; any later call with the same
inputs reads the saved file instead. `inputs` is everything the result depends on: the
feature sets, the model class, and the tables, by a fingerprint of their contents
(`table_digest`). So a change to any of them is a new fingerprint and a fresh run, and an
old result can never be read back for new inputs.

**What a fingerprint cannot see is a change to the code** that makes the result -- the
model, the cross-validation, the comparison. After changing that, pass `rerun=True`.

The files go to `reports/_cache/answer_hidden/` (`config/outputs.py::CACHE_ROOT`). Each
result is one pickle beside a small JSON naming what it was made from. Results made from
inputs no longer in use are left behind; delete the folder to clear them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

import pandas as pd

from src.config.outputs import CACHE_ROOT

CACHE_DIR = CACHE_ROOT / "answer_hidden"


def table_digest(df: pd.DataFrame) -> str:
    """A fingerprint of a table's contents: its column names and every value, in order."""
    h = hashlib.sha1()
    h.update("\x1f".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def cached(
    name: str,
    inputs: Mapping[str, Any],
    compute: Callable[[], Any],
    *,
    rerun: bool = False,
    verbose: bool = True,
) -> Any:
    """`compute()`, saved under `name` and the fingerprint of `inputs`, or the saved result
    of an earlier call with the same inputs. `inputs` must be JSON-serialisable: lists of
    columns, names, numbers, and tables as `table_digest`s. `rerun=True` computes again
    and replaces the saved result."""
    blob = json.dumps(inputs, sort_keys=True)
    path = CACHE_DIR / f"{name}__{hashlib.sha1(blob.encode()).hexdigest()[:12]}.pkl"
    if path.exists() and not rerun:
        if verbose:
            print(f"[cache] {name}: read {path.name}")
        return pd.read_pickle(path)

    result = compute()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, path)
    path.with_suffix(".json").write_text(json.dumps(inputs, sort_keys=True, indent=1), encoding="utf-8")
    if verbose:
        print(f"[cache] {name}: saved {path.name}")
    return result
