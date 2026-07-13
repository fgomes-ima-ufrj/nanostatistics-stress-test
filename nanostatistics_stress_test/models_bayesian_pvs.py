"""PyMC implementation of the PVS-aware mechanistically informed workflow."""
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
from .pvs import flatten_pymc_posterior, predictive_from_idata, pvs_descriptor, pvs_joint, pvs_pred, pvs_theta_from_samples


def _prior_settings(scenario: ScenarioConfig, area_median: float) -> Dict[str, float]:
    """Scenario-specific prior settings for sensitivity analysis."""
    mode = scenario.prior_mode
    if mode == "narrow":
        return {"ymax_sigma": 0.35, "K_sigma": 0.35, "h_sigma": 0.25, "sigma_scale": 0.10, "K_mu": np.log(area_median), "h_mu": np.log(max(scenario.h, 1.0))}
    if mode == "broad":
        return {"ymax_sigma": 1.50, "K_sigma": 1.25, "h_sigma": 0.90, "sigma_scale": 0.50, "K_mu": np.log(area_median), "h_mu": np.log(2.0)}
    if mode == "mildly_misspecified":
        return {"ymax_sigma": 0.80, "K_sigma": 0.60, "h_sigma": 0.50, "sigma_scale": 0.25, "K_mu": np.log(area_median * 1.8), "h_mu": np.log(1.2)}
    return {"ymax_sigma": 0.80, "K_sigma": 0.70, "h_sigma": 0.50, "sigma_scale": 0.25, "K_mu": np.log(area_median), "h_mu": np.log(2.5)}


def fit_pvs_pymc(
    df: pd.DataFrame,
    config: DGPConfig,
    inference: InferenceConfig,
    scenario: ScenarioConfig,
    simulation_id: int,
    seed: int,
    trace_path=None,
) -> Tuple[Dict[str, object], object]:
    """Fit the PVS-aware Hill/area hierarchical Bayesian model with Student-t residuals."""
    start = time.perf_counter()
    row = base_metric_row(simulation_id, scenario.scenario_id, "PVS_aware_area_MCMC", "reactive_area", df)
    try:
        import pymc as pm

        y = df["y"].to_numpy(dtype=float)
        area = df["area"].to_numpy(dtype=float)
        lab_idx = df["lab_idx"].to_numpy(dtype=int)
        batch_idx = df["batch_idx"].to_numpy(dtype=int)
        n_labs = int(df["lab_idx"].nunique())
        n_batches = int(df["batch_idx"].nunique())
        coords = {"obs": np.arange(len(df)), "lab": np.arange(n_labs), "batch": np.arange(n_batches)}
        pri = _prior_settings(scenario, float(np.median(area)))

        with pm.Model(coords=coords) as model:
            pm.Data("area", area, dims="obs")
            pm.Data("lab_idx", lab_idx, dims="obs")
            pm.Data("batch_idx", batch_idx, dims="obs")
            y0 = pm.TruncatedNormal("y0", mu=0.05, sigma=0.20, lower=-0.10, upper=0.30)
            ymax = pm.HalfNormal("ymax", sigma=pri["ymax_sigma"])
            K_A = pm.LogNormal("K_A", mu=pri["K_mu"], sigma=pri["K_sigma"])
            h = pm.LogNormal("h", mu=pri["h_mu"], sigma=pri["h_sigma"])
            sigma0 = pm.HalfNormal("sigma0", sigma=pri["sigma_scale"])
            rho_A = pm.HalfNormal("rho_A", sigma=0.5)
            sigma_lab = pm.HalfNormal("sigma_lab", sigma=0.2)
            sigma_batch = pm.HalfNormal("sigma_batch", sigma=0.2)
            z_lab = pm.Normal("z_lab", mu=0.0, sigma=1.0, dims="lab")
            z_batch = pm.Normal("z_batch", mu=0.0, sigma=1.0, dims="batch")
            a_lab = pm.Deterministic("a_lab", z_lab * sigma_lab, dims="lab")
            c_batch = pm.Deterministic("c_batch", z_batch * sigma_batch, dims="batch")
            nu = pm.Deterministic("nu", pm.Exponential("nu_minus_two", 1 / 10) + 2.0)
            A = model["area"]
            mu_base = y0 + ymax * (A**h) / (K_A**h + A**h)
            mu = mu_base + a_lab[model["lab_idx"]] + c_batch[model["batch_idx"]]
            sigma = sigma0 * (1.0 + rho_A * A)
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
        samples = flatten_pymc_posterior(idata, ["y0", "ymax", "K_A", "h", "sigma0", "rho_A", "sigma_lab", "sigma_batch", "nu"])
        theta_pvs, theta_ok = pvs_theta_from_samples(samples, config.y_phys_min, config.y_phys_max)
        pred_pvs = pvs_pred(pp, config.y_phys_min, config.y_phys_max)
        joint = pvs_joint(theta_ok, pp, config.y_phys_min, config.y_phys_max)
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
                "pvs_theta": theta_pvs,
                "pvs_pred": pred_pvs,
                "pvs_descriptor": pvs_descriptor(area),
                "pvs_joint": joint,
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
