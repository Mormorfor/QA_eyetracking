"""Parked work: real analyses that do not back the paper.

Kept rather than deleted (`docs/conventions.md`, "Keep, don't delete"), and kept
apart, so the path says which code produced a number in the paper and which did not:

* `src/analyses/` -- the paper's code.
* `src/explorations/` -- ran and produced something, but is not claimed in the paper.
  Some of it may be picked up again as future work. Some of it no longer runs; each
  folder's `__init__.py` says what state it is in.
* `archive/` -- superseded, and not expected to run.

Nothing in the paper's code imports from here.
"""
