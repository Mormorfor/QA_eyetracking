# src/vendor/julia_integration

`julia_test.ipynb` sets up Julia for this project and checks that Python can talk to it. The
only code that needs Julia is the parked mixed-model backend,
`src/explorations/mixed_models/models.py`, which fits its models with Julia's MixedModels
through `juliacall`. Nothing the paper uses needs it.

The notebook, top to bottom:

1. Installs Julia 1.12.4 through `juliapkg`, and adds the Julia packages the backend loads
   (MixedModels, GLM, DataFrames, Distributions, StatsBase, RCall).
2. Fits a mixed model on made-up data (2,000 rows, 30 subjects, 20 texts) and reads its
   coefficient table back into Python, to show that a fit runs end to end.
3. Tries ways of handing a pandas table to Julia. `jlconvert(jl.PyTable, df)` fails;
   wrapping it as `PythonCall.PyPandasDataFrame(df)` and building a Julia `DataFrame` from
   that works. `models.py` uses neither: it passes the columns across one at a time
   (`pycols_to_df`).

Run it once on a new machine before using the mixed-model backend. `juliacall` itself is
pinned in `environment.yml`.
