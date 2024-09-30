# Code for "Fiscal and Monetary Policy with Heterogeneous Agents"
This repository has replication code for ["Fiscal and Monetary Policy with Heterogeneous Agents"](https://web.stanford.edu/~aauclert/annual_review.pdf) (Auclert, Rognlie, Straub 2024), a draft article prepared for the Annual Review of Economics. 

The code is in Python and requires installing the [sequence-space Jacobian toolkit](https://github.com/shade-econ/sequence-jacobian).

The code is relatively compact, and is split between two Jupyter notebooks and a Python module:

1. `Annual Review hh calibration.ipynb` calibrates the baseline heterogeneous-agent model, producing Figures 1a and 1b and storing the calibrated parameters in `hh_params.json`.

2. `Annual Review main.ipynb` does our main fiscal and monetary policy experiments, producing Figures 2a, 2b, 3a, 3c, 4a, and 4b. (As long as `hh_params.json` is available, it can be run without re-running the first notebook.)

3. `household.py` is a support module used by both notebooks, specifying the household blocks and the exogenously-calibrated parameters for the HA, TA, and RA household models.

Two calibration inputs are in `inputs/`: a raw Lorenz curve extracted from SCF 2019 microdata (which we target in our calibration) and the Kaplan, Moll, Violante (2018) income process (which we use in `household.py` after converting to discrete time). All figures are saved to `figures/`.

We encourage anyone interested in coding HANK models to take a look at the code, especially `Annual Review main.ipynb`, where we consider many variations on fiscal and monetary policy experiments—including different policy response functions, different types of government debt, different incidence of fiscal shocks, cyclical income risk, households lacking rational expectations, and so on. Most of these variations only require changing a few lines of code.

It is also straightforward to adapt the code to consider different shocks. For instance, if we change the line specifying the monetary policy shock (in part 4 of the main notebook) from `dr = -0.25 * 0.9**np.arange(T)` to `dr = -0.25 * (np.arange(T)==10)`, the notebook then gives results for a "forward guidance" shock: in this case, a promise of a one-time 1% (annualized) cut in the real interest rate in 10 quarters.
