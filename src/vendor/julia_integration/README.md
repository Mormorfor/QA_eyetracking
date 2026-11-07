# src/vendor/julia_integration

`julia_test.ipynb` sets up Julia for this project and checks that Python can talk to it. The
only code that needs Julia is the mixed model, `src/modeling/models/glmm_model.py`, which fits
with Julia's MixedModels through `juliacall`, and what runs it
(`src/analyses/correctness_prediction/mixed_model/`). The logistic regression does not need it.

The notebook, top to bottom:

1. Declares Julia 1.12.4 and the Julia packages the backend loads (MixedModels, DataFrames,
   StatsModels, Distributions, StatsBase), at the versions last checked, and has `juliapkg`
   install them. They are declared to `juliapkg` rather than added with `Pkg.add`: when
   `juliapkg` re-resolves, it can rewrite the project's `Project.toml` from its own list, and
   a package it was not told about disappears.
2. Loads them and prints the versions.
3. Fits a mixed model on made-up data (2,000 rows, 30 subjects, 20 texts) and reads its
   coefficient table back into Python, to show that a fit runs end to end.
4. Tries ways of handing a pandas table to Julia. `jlconvert(jl.PyTable, df)` fails;
   wrapping it as `PythonCall.PyPandasDataFrame(df)` and building a Julia `DataFrame` from
   that works. `glmm_model.py` uses neither: it passes the columns across one at a time
   (`pycols_to_df`).

Run it once on a new machine before using the mixed model. `juliacall` itself is
pinned in `environment.yml`.

**Where the Julia packages live.** `juliapkg` keeps them in `<conda env>/julia_env/`, but it
finds the env through `CONDA_PREFIX`, which is set only when the env is *activated*. Run the
env's `python.exe` directly, or a Jupyter started from another env, and it would use
`~/.julia/environments/pyjuliapkg/` instead: a different, empty project, where it installs a
fresh Julia and then fails with `Package DataFrames not found`. `glmm_model.py` and the first
cell of the notebook both set `PYTHON_JULIAPKG_PROJECT` to `<this Python's env>/julia_env`
before `juliacall` is imported, so the folder follows the interpreter.
