# What does curryb-benchmarks have to do?

- We read in data files and produce `data.csv` and `options.csv` that currybo can read and understand
- We need a main loop somewhere that does the chemists' job
- That loop should call currybo, compare the result with actual data and compute the generality value
- A successful benchmark would see a growing generality over the number of iterations

## Generality

- given the optimum found by currybo, run the aggregation function over this crosssection and compare it with the actual aggregated optimum
