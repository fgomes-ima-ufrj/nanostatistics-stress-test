# Nanostatistics Synthetic Stress Test

Synthetic Monte Carlo benchmark for comparing three inferential workflows under nanoscale-like stress conditions:

1. OLS/ANOVA mass-based baseline.
2. Hierarchical Bayesian mass-based model.
3. PVS-aware mechanistically informed area-based model.

The data-generating process creates a descriptor mismatch: the observed response is governed by reactive surface area, while conventional baselines use nominal mass. The DGP includes laboratory effects, batch effects, a Hill-type saturating response, Student-t heavy-tailed noise, and area-dependent heteroscedasticity.

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fast development run using the Laplace fallback:

```bash
python run_stress_test.py --engine laplace_fast --n-sim 3 --output outputs_dev
```

Recommended PyMC development run:

```bash
python run_stress_test.py --engine pymc --n-sim 5 --chains 4 --tune 1000 --draws 1000 --target-accept 0.90 --output outputs_pymc_dev
```

Manuscript-scale run, if computationally feasible:

```bash
python run_stress_test.py --engine pymc --n-sim 100 --chains 4 --tune 2000 --draws 4000 --target-accept 0.95 --seed 20260630 --output outputs_manuscript
```

Sensitivity scenarios:

```bash
python run_stress_test.py --engine pymc --include-sweeps --n-sim 30 --chains 4 --tune 2000 --draws 4000 --target-accept 0.95 --output outputs_sweeps
```

## Output structure

The runner writes:

```text
outputs/
├── config_used.json
├── environment.txt
├── simulated_data/
├── posterior_traces/
├── model_summaries/
├── diagnostics/
├── tables/
│   ├── metrics_raw.csv
│   ├── summary_median_iqr.csv
│   └── manuscript_numbers.json
└── figures/
    ├── figure_S1_dgp.png
    ├── figure_S2_model_comparison.png
    └── figure_S3_pvs.png
```

## Interpretation notes

The Laplace engine is a fast approximation for development and debugging. Manuscript-level claims about posterior mass and PVS should use the PyMC or Stan MCMC engines.

PVS is implemented as a diagnostic of posterior or posterior-predictive admissibility. It is not implemented as a universal acceptance threshold or as a replacement for convergence diagnostics, posterior predictive checks, residual diagnostics, prior sensitivity, or domain expertise.


## Archived data and posterior traces

Large reproducibility artifacts are archived on Zenodo:

https://doi.org/10.5281/zenodo.21344153

The Zenodo archive includes:

- `posterior_traces_mass_150_ta099.zip`
- `posterior_traces_pvs_150_ta099.zip`
- `synthetic_data_and_truth_150_ta099.zip`
- `final_tables_figures_metadata_150_ta099.zip`
- `SHA256SUMS_release_artifacts.txt`

Posterior trace files correspond to the final manuscript run in `outputs_manuscript_150_ta099`. They were verified by simulation identifier, workflow type, posterior dimensions, and divergent transition counts against `metrics_raw.csv`. The verification returned 300/300 matching trace files and zero mismatches.
