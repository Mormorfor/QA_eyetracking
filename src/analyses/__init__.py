"""The code behind the paper's results: one folder per question.

Each folder works out its numbers, tests them where there is a test, and draws
the figures. Every figure is saved with its numbers under `reports/<folder>/`,
by `lib/plotting/output.py::save_output`. `scripts/run_analysis.py` runs most
folders without a notebook.

Work that does not back the paper is in `explorations/`, laid out the same way.
Nothing else in `src/` imports from here, except two modules in
`explorations/` that reuse `correctness_prediction/run.py` and its plots.
"""
