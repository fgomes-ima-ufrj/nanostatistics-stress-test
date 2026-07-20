# Prior-sensitivity analysis

This directory contains the validated prior-sensitivity analysis conducted on
a reproducible random subset of 30 simulations from the final 150-simulation
Nanostatistics stress test.

## Selection design

The subset was selected using:

- selection seed: `20260716`
- selected simulations: `30`
- simulation-ID range: `0` to `149`

The exact selected IDs are recorded in:

`selected_simulation_ids.json`

## Evaluated prior specifications

Four prior modes were compared:

- nominal
- narrow
- broad
- mildly misspecified

The nominal results were recovered from the archived final manuscript run.

The three alternative prior specifications required 90 new Bayesian fits:

`30 simulations x 3 alternative priors = 90 new fits`

Each new fit used:

- PyMC/NUTS
- 4 chains
- 3000 tuning iterations per chain
- 3000 retained posterior draws per chain
- target acceptance probability of 0.99

## Paired analysis

Each alternative-prior result was compared with the nominal result from the
same synthetic dataset.

The analysis includes paired differences for:

- RMSE
- MAE
- posterior predictive coverage
- parameter-space PVS
- predictive PVS
- joint PVS
- divergences
- R-hat
- bulk and tail effective sample sizes
- BFMI
- runtime

Parameter-level posterior shifts relative to the nominal prior are also
provided.

## Main findings

Predictive performance was practically invariant across prior specifications.

Median paired RMSE differences relative to nominal were:

- narrow: `4.36e-7`
- broad: `-8.88e-6`
- mildly misspecified: `-6.36e-6`

Median paired predictive-PVS differences were:

- narrow: `0.000575`
- broad: `-0.000107`
- mildly misspecified: `-0.000249`

Median paired joint-PVS differences were:

- narrow: `0.004656`
- broad: `0.002476`
- mildly misspecified: `0.000338`

These results support practical robustness to the evaluated prior
specifications rather than mathematical invariance.

## MCMC diagnostics

All 120 compared fits satisfied the adopted thresholds for:

- R-hat
- bulk effective sample size
- tail effective sample size
- BFMI

Divergences occurred in 25 of the 120 fits and were concentrated in 10 of the
30 simulation datasets.

The number of fits with divergences was:

- nominal: 6 of 30
- narrow: 7 of 30
- broad: 6 of 30
- mildly misspecified: 6 of 30

A conservative paired analysis removed every simulation that presented a
divergence under any prior specification. The resulting set contained 20
fully divergence-free simulations and retained the same substantive
conclusion.

## Interpretation

This analysis used a fixed random subset of 30 of the 150 simulations. It
should be interpreted as a targeted robustness analysis rather than an
exhaustive rerun of the complete simulation corpus.

The alternative priors did not materially change predictive performance,
predictive admissibility, or the overall inferential interpretation.

## Trace files

The 90 NetCDF posterior trace files are included in the `traces` directory and
are managed through Git Large File Storage.

The collection contains:

- 30 traces for the narrow prior;
- 30 traces for the broad prior;
- 30 traces for the mildly misspecified prior.

The nominal traces were recovered from the previously archived final
simulation run and are not duplicated in this directory.

File-level SHA-256 hashes for all 90 NetCDF files are provided in:

`SHA256SUMS_prior_sensitivity_traces_n30.txt`

Git LFS must be installed before cloning or pulling the repository to retrieve
the actual NetCDF files rather than only the LFS pointer objects.

## Directory contents

The `tables` directory includes:

- raw fit-level metrics;
- aggregated summaries;
- paired differences relative to nominal;
- divergence-free sensitivity summaries;
- MCMC diagnostic summaries;
- divergence concentration by simulation;
- raw parameter summaries;
- paired parameter shifts.

## Integrity

File-level SHA-256 hashes for the files included in this directory are
provided in:

`SHA256SUMS_prior_sensitivity_n30_ta099.txt`

## Conservative zero-divergence subset

The conservative paired subset was defined by excluding every simulation
with one or more post-tuning divergent transitions under any of the four
evaluated prior specifications: nominal, narrow, broad, and mildly
misspecified.

The original prior-sensitivity subset comprised 30 simulations selected
without replacement using selection seed 20260716. Twenty simulations met
the conservative retention criterion.

The retained identifiers are provided in:

`tables/conservative_zero_divergence_subset_n20.csv`

A human-readable record is provided in:

`tables/conservative_zero_divergence_subset_n20.txt`

The complete 30-simulation filtering record is provided in:

`tables/prior_sensitivity_divergence_filter_n30.csv`

The excluded simulations and their prior-specific divergence burdens are
provided in:

`tables/excluded_divergent_subset_n10.csv`

The associated integrity record is:

`tables/SHA256SUMS_conservative_subset_n20.txt`

The subset was derived directly from the validated simulation-by-prior
diagnostic output. No model refitting, manual identifier selection, or
trace-level inference was required.
