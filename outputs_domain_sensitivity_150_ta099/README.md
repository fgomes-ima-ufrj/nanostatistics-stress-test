# Predictive-domain sensitivity analysis

This directory contains the validated predictive-domain sensitivity analysis
recomputed from the archived posterior and posterior predictive traces of the
final 150-simulation manuscript run.

## Source traces

The analysis used:

- `posterior_traces_mass_150_ta099.zip`
- `posterior_traces_pvs_150_ta099.zip`

Both archives are available in the associated Zenodo release:

- DOI: `10.5281/zenodo.21344153`

## Evaluated predictive domains

- Narrow: `[0.02, 0.98]`
- Nominal: `[0.00, 1.00]`
- Broad: `[-0.05, 1.05]`

## Analysis design

The archived traces contain:

- 150 simulations per Bayesian workflow
- 4 chains per simulation
- 3000 retained posterior draws per chain
- 288 posterior predictive outcomes per posterior draw

No new MCMC sampling was performed for this domain-sensitivity analysis.

For the mass-based Bayesian workflow, only predictive PVS was recalculated.
Parameter PVS and joint PVS remain undefined because no mechanistic parameter
admissibility domain was specified for this workflow.

For the area-based workflow, parameter PVS, predictive PVS, and joint PVS were
recalculated under all three domains.

## Validation

The nominal-domain recalculation reproduced the archived values for all 150
simulations of each applicable metric.

- Maximum absolute difference for mass-based predictive PVS: `0`
- Maximum absolute difference for area-based PVS quantities:
  approximately `1.11e-16`
- Differences greater than `1e-12`: `0`
- Differences greater than `1e-8`: `0`

All 300 workflow-simulation pairs satisfied:

`PVS_narrow <= PVS_nominal <= PVS_broad`

## Main median results

| Workflow | Domain | Parameter PVS | Predictive PVS | Joint PVS |
|---|---:|---:|---:|---:|
| Area-based | Narrow | 0.592750 | 0.836109 | 0.487773 |
| Area-based | Nominal | 0.822708 | 0.906879 | 0.743193 |
| Area-based | Broad | 0.967083 | 0.980696 | 0.948913 |
| Mass-based | Narrow | NA | 0.752325 | NA |
| Mass-based | Nominal | NA | 0.773553 | NA |
| Mass-based | Broad | NA | 0.822476 | NA |

## Interpretation

This sensitivity analysis does not identify an optimal physical domain.
It quantifies the expected dependence of PVS on explicitly narrower or broader
admissibility definitions.

The relative predictive advantage of the area-based workflow over the
mass-based workflow was preserved across all three domains.

## Output structure

The directory contains:

- raw simulation-level recalculations;
- aggregated median and interquartile-range summaries;
- paired differences relative to the nominal domain;
- nominal recalculation validation;
- completeness checks;
- manuscript-ready summary table;
- diagnostic figures;
- trace-archive inspection metadata;
- file-level SHA-256 checksums.

## Integrity

File-level SHA-256 hashes are provided in:

`SHA256SUMS_domain_sensitivity_150_ta099.txt`
