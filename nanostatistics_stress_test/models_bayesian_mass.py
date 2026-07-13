"""PyMC implementation of the hierarchical mass-based Bayesian workflow."""
from __future__ import annotations

from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd

from .config import DGPConfig, InferenceConfig, ScenarioConfig
from .diagnostics import (
    base_metric_row,
    diagnostic_flag,
    interval_coverage,
    mae,
    predictive_mean,
    residual_diagnostics,
    r2_score,
    rmse,
    summarize_arviz,
    transition_region_error,
    variance_recovery,
)
from .pvs import predictive_from_idata, pvs_descriptor, pvs_pred


def fit_mass_pymc(
    df: pd.DataFrame,
    config: DGPConfig,
    inference: InferenceConfig,
    scenario: ScenarioConfig,
    simulation_id: int,
    seed: int,
    trace_path=None,
) -> Tuple[Dict[str, object], object]:
    """Fit the mass-based hierarchical Bayesian model with Student-t residuals."""
    start = time.perf_counter()
    row = base_metric_row(simulation_id, scenario.scenario_id, "Bayes_hierarchical_mass_MCMC", "mass", df)
    try:
        import pymc as pm
        import arviz as az

        y = df["y"].to_numpy(dtype=float)
        x = df["log_mass_z"].to_numpy(dtype=float)
        lab_idx = df["lab_idx"].to_numpy(dtype=int)
        batch_idx = df["batch_idx"].to_numpy(dtype=int)
        n_labs = int(df["lab_idx"].nunique())
        n_batches = int(df["batch_idx"].nunique())
        coords = {"obs": np.arange(len(df)), "lab": np.arange(n_labs), "batch": np.arange(n_batches)}

        with pm.Model(coords=coords) as model:
            pm.Data("x", x, dims="obs")
            pm.Data("lab_idx", lab_idx, dims="obs")
            pm.Data("batch_idx", batch_idx, dims="obs")
            alpha = pm.Normal("alpha", mu=float(np.mean(y)), sigma=1.0)
            beta_m = pm.Normal("beta_m", mu=0.0, sigma=1.0)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            sigma_lab = pm.HalfNormal("sigma_lab", sigma=0.2)
            sigma_batch = pm.HalfNormal("sigma_batch", sigma=0.2)
            z_lab = pm.Normal("z_lab", mu=0.0, sigma=1.0, dims="lab")
            z_batch = pm.Normal("z_batch", mu=0.0, sigma=1.0, dims="batch")
            a_lab = pm.Deterministic("a_lab", z_lab * sigma_lab, dims="lab")
            c_batch = pm.Deterministic("c_batch", z_batch * sigma_batch, dims="batch")
            nu = pm.Deterministic("nu", pm.Exponential("nu_minus_two", 1 / 10) + 2.0)
            mu = alpha + beta_m * model["x"] + a_lab[model["lab_idx"]] + c_batch[model["batch_idx"]]
            pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y, dims="obs")
            idata = pm.sample(
                draws=inference.draws,
                tune=inference.tune,
                chains=inference.chains,
                cores=inference.cores,
                target_accept=inference.target_accept,
                random_seed=seed,
                progressbar=inference.progressbar,
                return_inferencedata=True,
            )
            idata = pm.sample_posterior_predictive(idata, random_seed=seed + 1, progressbar=inference.progressbar, extend_inferencedata=True)

        if trace_path is not None and inference.save_traces:
            idata.to_netcdf(str(trace_path))

        pp = predictive_from_idata(idata, "y_obs")
        y_hat = predictive_mean(pp)
        rd = residual_diagnostics(y, y_hat, design=df[["log_mass_z", "area"]].to_numpy())
        mcmc = summarize_arviz(idata)
        pred_pvs = pvs_pred(pp, config.y_phys_min, config.y_phys_max)
        row.update(
            {
                "rmse": rmse(y, y_hat),
                "mae": mae(y, y_hat),
                "r2_or_pseudo_r2": r2_score(y, y_hat),
                "aic": np.nan,
                "bic": np.nan,
                "coverage_50": interval_coverage(y, pp, 0.50),
                "coverage_80": interval_coverage(y, pp, 0.80),
                "coverage_90": interval_coverage(y, pp, 0.90),
                "coverage_95": interval_coverage(y, pp, 0.95),
                "ppc_coverage": interval_coverage(y, pp, 0.95),
                "lab_variance_recovery": variance_recovery(float(idata.posterior["sigma_lab"].mean().item()), config.sigma_lab_y),
                "batch_variance_recovery": variance_recovery(float(idata.posterior["sigma_batch"].mean().item()), config.sigma_batch_y),
                "transition_region_error": transition_region_error(df, y_hat),
                "extrapolation_error": np.nan,
                "pvs_theta": np.nan,
                "pvs_pred": pred_pvs,
                "pvs_descriptor": pvs_descriptor(df["area"].to_numpy()),
                "pvs_joint": pred_pvs,
                "prior_sensitivity_delta_pvs": np.nan,
                "admissibility_sensitivity_delta_pvs": np.nan,
                "runtime_seconds": time.perf_counter() - start,
            }
        )
        row.update(rd)
        row.update(mcmc)
        row["diagnostic_flag"] = diagnostic_flag(row)  # type: ignore[arg-type]
        return row, idata
    except Exception as exc:
        row.update({"diagnostic_flag": f"fit_failed:{type(exc).__name__}", "runtime_seconds": time.perf_counter() - start})
        return row, None
