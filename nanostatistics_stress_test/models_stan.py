"""Optional CmdStanPy implementation of the Bayesian workflows.

This module is complete but optional. It requires cmdstanpy and an installed CmdStan
 toolchain. PyMC is the preferred default MCMC engine for most users because ArviZ
 integration is simpler.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd

from .config import DGPConfig, InferenceConfig, ScenarioConfig
from .diagnostics import base_metric_row, diagnostic_flag, interval_coverage, mae, predictive_mean, residual_diagnostics, r2_score, rmse, transition_region_error, variance_recovery
from .pvs import pvs_descriptor, pvs_joint, pvs_pred, pvs_theta_from_samples


MASS_STAN = r'''
data {
  int<lower=1> N;
  int<lower=1> L;
  int<lower=1> B;
  vector[N] y;
  vector[N] x;
  array[N] int<lower=1, upper=L> lab;
  array[N] int<lower=1, upper=B> batch;
}
parameters {
  real alpha;
  real beta_m;
  real<lower=0> sigma;
  real<lower=0> sigma_lab;
  real<lower=0> sigma_batch;
  vector[L] z_lab;
  vector[B] z_batch;
  real<lower=0> nu_minus_two;
}
transformed parameters {
  vector[L] a_lab = z_lab * sigma_lab;
  vector[B] c_batch = z_batch * sigma_batch;
  real<lower=2> nu = nu_minus_two + 2;
}
model {
  alpha ~ normal(mean(y), 1);
  beta_m ~ normal(0, 1);
  sigma ~ normal(0, 0.5);
  sigma_lab ~ normal(0, 0.2);
  sigma_batch ~ normal(0, 0.2);
  z_lab ~ normal(0, 1);
  z_batch ~ normal(0, 1);
  nu_minus_two ~ exponential(0.1);
  for (n in 1:N) {
    y[n] ~ student_t(nu, alpha + beta_m * x[n] + a_lab[lab[n]] + c_batch[batch[n]], sigma);
  }
}
generated quantities {
  vector[N] y_rep;
  vector[N] mu;
  for (n in 1:N) {
    mu[n] = alpha + beta_m * x[n] + a_lab[lab[n]] + c_batch[batch[n]];
    y_rep[n] = student_t_rng(nu, mu[n], sigma);
  }
}
'''

PVS_STAN = r'''
data {
  int<lower=1> N;
  int<lower=1> L;
  int<lower=1> B;
  vector[N] y;
  vector<lower=0>[N] area;
  real<lower=0> area_median;
  array[N] int<lower=1, upper=L> lab;
  array[N] int<lower=1, upper=B> batch;
  real<lower=0> y_phys_min;
  real y_phys_max;
}
parameters {
  real<lower=-0.10, upper=0.30> y0;
  real<lower=0> ymax;
  real<lower=0> K_A;
  real<lower=0> h;
  real<lower=0> sigma0;
  real<lower=0> rho_A;
  real<lower=0> sigma_lab;
  real<lower=0> sigma_batch;
  vector[L] z_lab;
  vector[B] z_batch;
  real<lower=0> nu_minus_two;
}
transformed parameters {
  vector[L] a_lab = z_lab * sigma_lab;
  vector[B] c_batch = z_batch * sigma_batch;
  real<lower=2> nu = nu_minus_two + 2;
}
model {
  y0 ~ normal(0.05, 0.20);
  ymax ~ normal(0, 0.8);
  K_A ~ lognormal(log(area_median), 0.7);
  h ~ lognormal(log(2.5), 0.5);
  sigma0 ~ normal(0, 0.25);
  rho_A ~ normal(0, 0.5);
  sigma_lab ~ normal(0, 0.2);
  sigma_batch ~ normal(0, 0.2);
  z_lab ~ normal(0, 1);
  z_batch ~ normal(0, 1);
  nu_minus_two ~ exponential(0.1);
  for (n in 1:N) {
    real mu_base = y0 + ymax * pow(area[n], h) / (pow(K_A, h) + pow(area[n], h));
    real mu = mu_base + a_lab[lab[n]] + c_batch[batch[n]];
    real sigma = sigma0 * (1 + rho_A * area[n]);
    y[n] ~ student_t(nu, mu, sigma);
  }
}
generated quantities {
  vector[N] y_rep;
  vector[N] mu;
  int theta_admissible;
  theta_admissible = (y0 >= y_phys_min && ymax >= 0 && K_A > 0 && h > 0 && sigma0 > 0 && rho_A >= 0 && sigma_lab > 0 && sigma_batch > 0 && nu > 2 && (y0 + ymax) <= y_phys_max);
  for (n in 1:N) {
    real mu_base = y0 + ymax * pow(area[n], h) / (pow(K_A, h) + pow(area[n], h));
    real sigma = sigma0 * (1 + rho_A * area[n]);
    mu[n] = mu_base + a_lab[lab[n]] + c_batch[batch[n]];
    y_rep[n] = student_t_rng(nu, mu[n], sigma);
  }
}
'''


def _compile_model(code: str, name: str, model_dir: Path):
    from cmdstanpy import CmdStanModel
    model_dir.mkdir(parents=True, exist_ok=True)
    stan_file = model_dir / f"{name}.stan"
    stan_file.write_text(code, encoding="utf-8")
    return CmdStanModel(stan_file=str(stan_file))


def _fit_stan_model(code: str, name: str, data: Dict, inference: InferenceConfig, seed: int, model_dir: Path):
    model = _compile_model(code, name, model_dir)
    return model.sample(
        data=data,
        chains=inference.chains,
        iter_warmup=inference.tune,
        iter_sampling=inference.draws,
        seed=seed,
        adapt_delta=inference.target_accept,
        show_progress=inference.progressbar,
    )


def fit_mass_stan(df: pd.DataFrame, config: DGPConfig, inference: InferenceConfig, scenario: ScenarioConfig, simulation_id: int, seed: int, model_dir: Path) -> Tuple[Dict[str, object], object]:
    start = time.perf_counter()
    row = base_metric_row(simulation_id, scenario.scenario_id, "Bayes_hierarchical_mass_Stan", "mass", df)
    try:
        data = {
            "N": len(df),
            "L": int(df["lab_idx"].nunique()),
            "B": int(df["batch_idx"].nunique()),
            "y": df["y"].to_numpy(dtype=float),
            "x": df["log_mass_z"].to_numpy(dtype=float),
            "lab": (df["lab_idx"].to_numpy(dtype=int) + 1),
            "batch": (df["batch_idx"].to_numpy(dtype=int) + 1),
        }
        fit = _fit_stan_model(MASS_STAN, "mass_hierarchical", data, inference, seed, model_dir)
        y_rep_cols = [c for c in fit.stan_variables().keys() if c == "y_rep"]
        pp = fit.stan_variable("y_rep")
        y_hat = predictive_mean(pp)
        rd = residual_diagnostics(data["y"], y_hat, design=df[["log_mass_z", "area"]].to_numpy())
        pred_pvs = pvs_pred(pp, config.y_phys_min, config.y_phys_max)
        row.update(
            {
                "rmse": rmse(data["y"], y_hat), "mae": mae(data["y"], y_hat), "r2_or_pseudo_r2": r2_score(data["y"], y_hat),
                "coverage_50": interval_coverage(data["y"], pp, 0.50), "coverage_80": interval_coverage(data["y"], pp, 0.80),
                "coverage_90": interval_coverage(data["y"], pp, 0.90), "coverage_95": interval_coverage(data["y"], pp, 0.95),
                "ppc_coverage": interval_coverage(data["y"], pp, 0.95), "pvs_pred": pred_pvs,
                "pvs_descriptor": pvs_descriptor(df["area"].to_numpy()), "pvs_joint": pred_pvs,
                "pvs_theta": np.nan, "transition_region_error": transition_region_error(df, y_hat),
                "lab_variance_recovery": variance_recovery(float(np.mean(fit.stan_variable("sigma_lab"))), config.sigma_lab_y),
                "batch_variance_recovery": variance_recovery(float(np.mean(fit.stan_variable("sigma_batch"))), config.sigma_batch_y),
                "rhat_max": np.nan, "ess_bulk_min": np.nan, "ess_tail_min": np.nan, "n_divergences": np.nan, "bfmi_min": np.nan,
                "aic": np.nan, "bic": np.nan, "extrapolation_error": np.nan, "prior_sensitivity_delta_pvs": np.nan, "admissibility_sensitivity_delta_pvs": np.nan,
                "runtime_seconds": time.perf_counter() - start,
            }
        )
        row.update(rd)
        row["diagnostic_flag"] = diagnostic_flag(row)  # type: ignore[arg-type]
        return row, fit
    except Exception as exc:
        row.update({"diagnostic_flag": f"fit_failed:{type(exc).__name__}", "runtime_seconds": time.perf_counter() - start})
        return row, None


def fit_pvs_stan(df: pd.DataFrame, config: DGPConfig, inference: InferenceConfig, scenario: ScenarioConfig, simulation_id: int, seed: int, model_dir: Path) -> Tuple[Dict[str, object], object]:
    start = time.perf_counter()
    row = base_metric_row(simulation_id, scenario.scenario_id, "PVS_aware_area_Stan", "reactive_area", df)
    try:
        data = {
            "N": len(df), "L": int(df["lab_idx"].nunique()), "B": int(df["batch_idx"].nunique()),
            "y": df["y"].to_numpy(dtype=float), "area": df["area"].to_numpy(dtype=float),
            "lab": (df["lab_idx"].to_numpy(dtype=int) + 1), "batch": (df["batch_idx"].to_numpy(dtype=int) + 1),
            "y_phys_min": config.y_phys_min, "y_phys_max": config.y_phys_max, "area_median": float(np.median(df["area"].to_numpy(dtype=float))),
        }
        fit = _fit_stan_model(PVS_STAN, "pvs_area_hill", data, inference, seed, model_dir)
        pp = fit.stan_variable("y_rep")
        y_hat = predictive_mean(pp)
        samples = {k: fit.stan_variable(k) for k in ["y0", "ymax", "K_A", "h", "sigma0", "rho_A", "sigma_lab", "sigma_batch", "nu"]}
        theta_pvs, theta_ok = pvs_theta_from_samples(samples, config.y_phys_min, config.y_phys_max)
        pred_pvs = pvs_pred(pp, config.y_phys_min, config.y_phys_max)
        rd = residual_diagnostics(data["y"], y_hat, design=df[["log_mass_z", "area"]].to_numpy())
        row.update(
            {
                "rmse": rmse(data["y"], y_hat), "mae": mae(data["y"], y_hat), "r2_or_pseudo_r2": r2_score(data["y"], y_hat),
                "coverage_50": interval_coverage(data["y"], pp, 0.50), "coverage_80": interval_coverage(data["y"], pp, 0.80),
                "coverage_90": interval_coverage(data["y"], pp, 0.90), "coverage_95": interval_coverage(data["y"], pp, 0.95),
                "ppc_coverage": interval_coverage(data["y"], pp, 0.95), "pvs_theta": theta_pvs, "pvs_pred": pred_pvs,
                "pvs_descriptor": pvs_descriptor(df["area"].to_numpy()), "pvs_joint": pvs_joint(theta_ok, pp, config.y_phys_min, config.y_phys_max),
                "transition_region_error": transition_region_error(df, y_hat),
                "lab_variance_recovery": variance_recovery(float(np.mean(fit.stan_variable("sigma_lab"))), config.sigma_lab_y),
                "batch_variance_recovery": variance_recovery(float(np.mean(fit.stan_variable("sigma_batch"))), config.sigma_batch_y),
                "rhat_max": np.nan, "ess_bulk_min": np.nan, "ess_tail_min": np.nan, "n_divergences": np.nan, "bfmi_min": np.nan,
                "aic": np.nan, "bic": np.nan, "extrapolation_error": np.nan, "prior_sensitivity_delta_pvs": np.nan, "admissibility_sensitivity_delta_pvs": np.nan,
                "runtime_seconds": time.perf_counter() - start,
            }
        )
        row.update(rd)
        row["diagnostic_flag"] = diagnostic_flag(row)  # type: ignore[arg-type]
        return row, fit
    except Exception as exc:
        row.update({"diagnostic_flag": f"fit_failed:{type(exc).__name__}", "runtime_seconds": time.perf_counter() - start})
        return row, None
