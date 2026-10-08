![CurryBO-logo](https://raw.githubusercontent.com/digital-chemistry-laboratory/currybo/refs/heads/main/CurryBO-logo.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)

[![arXiv](https://img.shields.io/badge/arXiv-2502.18966-b31b1b.svg)](https://arxiv.org/abs/2502.18966)



# Benchmarking for CurryBO: Bayesian optimization over curried function spaces

`CurryBO_benchmarks` is a package that allows to run the chemical reaction optimization benchmarks for the [CurryBO paper](https://arxiv.org/abs/2502.18966), which can be found in the [currybo repository](https://github.com/digital-chemistry-laboratory/currybo).

## Installation

To install the package, simply:
```
pip install currybo-benchmarks
```

## Usage

If you are interested in applying the CurryBO benchmarks for chemical reaction optimization outlined in the CurryBO paper, you can simply call
```
currybo-benchmarks [Denmark/Deoxyfluorination/Cernak/Borylation]
```

Denmark/Deoxyfluronation/Cernak/Borylation are the respective datasets discussed in the paper, that you can also find in this repository under `src/currybo_benchmarks/datasets/chemistry_datasets/`. 
This script will output a summary of the run, the expected generality for each condition, and the observed generality and GAP values (what is plotted in the paper) at each iteration. Additionally, you can set more arguments for modifying your CurryBO settings, as explained in the [documentation](https://currybo.ethz.ch/docs). 
By setting the arguments as specified in the paper, the plots can be reproduced.

## Citation

If you use CurryBO in your research, please cite the corresponding paper:

```
@misc{schmid_one_2025,
	title = {One {Set} to {Rule} {Them} {All}: {How} to {Obtain} {General} {Chemical} {Conditions} via {Bayesian} {Optimization} over {Curried} {Functions}},
	url = {http://arxiv.org/abs/2502.18966},
	doi = {10.48550/arXiv.2502.18966},
	publisher = {arXiv},
	author = {Schmid, Stefan P. and Rajaonson, Ella Miray and Ser, Cher Tian and Haddadnia, Mohammad and Leong, Shi Xuan and Aspuru-Guzik, Alán and Kristiadi, Agustinus and Jorner, Kjell and Strieth-Kalthoff, Felix},
	month = feb,
	year = {2025},
	note = {arXiv:2502.18966 [cs]},
}
```
