"""Fast approximate Bayesian fallback using MAP/MLE plus Gaussian posterior draws.

This module is intended for development, triage, and supplementary checks when MCMC
is unavailable or too costly. It is not the preferred engine for manuscript-scale PVS
claims, which should use PyMC or Stan.
"""
from __future__ import annotations

from typing import Dict, Tuple
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

from .config import DGPConfig, ScenarioConfig
from .diagnostics import (
    base_metric_row,
    diagnostic_flag,
    interval_coverage,
    mae,
    predictive_mean,
    residual_diagnostics,
    r2_score,
    rmse,
    transition_region_error,
    variance_recovery,
)
from .dgp import hill_response
from .pvs import pvs_descriptor, pvs_joint, pvs_pred, pvs_theta_from_samples


def _safe_cov_from_hess_inv(hess_inv, n: int) -> np.ndarray:
    if hasattr(hess_inv, "todense"):
        cov = np.asarray(hess_inv.todense(), dtype=float)
    else:
        cov = np.asarray(hess_inv, dtype=float)
    if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
        cov = np.eye(n) * 0.05
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = np.min(eigvals)
    if min_eig <= 1e-10:
        cov += np.eye(n) * (abs(min_eig) + 1e-6)
    return cov


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def fit_mass_laplace(
    df: pd.DataFrame,
    config: DGPConfig,
    scenario: ScenarioConfig,
    simulation_id: int,
    seed: int,
    n_draws: int = 2000,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Approximate hierarchical mass-based Bayesian model with fixed effects.

    Uses OLS covariance to draw approximate coefficient samples and residual draws to
    mimic posterior predictive uncertainty. This preserves the descriptor mismatch and
    provides fast PPC/PVS_pred diagnostics, but it is not a replacement for MCMC.
    """
    start = time.perf_counter()
    rng = _rng(seed)
    import statsmodels.formula.api as smf

    row = base_metric_row(simulation_id, scenario.scenario_id, "Bayes_hierarchical_mass_laplace_approx", "mass", df)
    try:
        model = smf.ols("y ~ log_mass_z + C(lab) + C(batch)", data=df).fit()
        X = model.model.exog
        beta_hat = np.asarray(model.params, dtype=float)
        cov_beta = np.asarray(model.cov_params(), dtype=float)
        cov_beta = 0.5 * (cov_beta + cov_beta.T)
        cov_beta += np.eye(cov_beta.shape[0]) * 1e-9
        beta_draws = rng.multivariate_normal(beta_hat, cov_beta, size=n_draws, method="svd")
        sigma_hat = float(np.sqrt(model.scale))
        sigma_draws = np.abs(rng.normal(sigma_hat, sigma_hat / np.sqrt(max(len(df), 2)), size=n_draws))
        nu = max(float(config.nu), 2.1)
        mu_draws = beta_draws @ X.T
        pp = mu_draws + sigma_draws[:, None] * rng.standard_t(df=nu, size=mu_draws.shape)
        y_hat = predictive_mean(pp)

        rd = residual_diagnostics(df["y"].to_numpy(), y_hat, design=df[["log_mass_z", "area"]].to_numpy())
        pred_pvs = pvs_pred(pp, config.y_phys_min, config.y_phys_max)
        row.update(
            {
                "rmse": rmse(df["y"].to_numpy(), y_hat),
                "mae": mae(df["y"].to_numpy(), y_hat),
                "r2_or_pseudo_r2": r2_score(df["y"].to_numpy(), y_hat),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "coverage_50": interval_coverage(df["y"].to_numpy(), pp, 0.50),
                "coverage_80": interval_coverage(df["y"].to_numpy(), pp, 0.80),
                "coverage_90": interval_coverage(df["y"].to_numpy(), pp, 0.90),
                "coverage_95": interval_coverage(df["y"].to_numpy(), pp, 0.95),
                "rhat_max": np.nan,
                "ess_bulk_min": np.nan,
                "ess_tail_min": np.nan,
                "n_divergences": np.nan,
                "bfmi_min": np.nan,
                "ppc_coverage": interval_coverage(df["y"].to_numpy(), pp, 0.95),
                "lab_variance_recovery": np.nan,
                "batch_variance_recovery": np.nan,
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
        row["diagnostic_flag"] = diagnostic_flag(row)  # type: ignore[arg-type]
        return row, {"posterior_predictive": pp, "mu_draws": mu_draws, "beta_draws": beta_draws}
    except Exception as exc:
        row.update({"diagnostic_flag": f"fit_failed:{type(exc).__name__}", "runtime_seconds": time.perf_counter() - start})
        return row, {}


def _unpack_hill_params(theta: np.ndarray, n_labs: int, n_batches: int) -> Dict[str, np.ndarray | float]:
    y0 = theta[0]
    ymax = np.exp(theta[1])
    K_A = np.exp(theta[2])
    h = np.exp(theta[3])
    sigma0 = np.exp(theta[4])
    rho_A = np.exp(theta[5])
    lab = theta[6 : 6 + n_labs]
    batch = theta[6 + n_labs : 6 + n_labs + n_batches]
    return {
        "y0": y0,
        "ymax": ymax,
        "K_A": K_A,
        "h": h,
        "sigma0": sigma0,
        "rho_A": rho_A,
        "lab_effect": lab,
        "batch_effect": batch,
        "sigma_lab": float(np.std(lab, ddof=1)) if n_labs > 1 else 1e-6,
        "sigma_batch": float(np.std(batch, ddof=1)) if n_batches > 1 else 1e-6,
    }


def fit_pvs_laplace(
    df: pd.DataFrame,
    config: DGPConfig,
    scenario: ScenarioConfig,
    simulation_id: int,
    seed: int,
    n_draws: int = 2000,
    maxiter: int = 2000,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Fast approximate PVS-aware Hill model using MAP plus Gaussian draws."""
    start = time.perf_counter()
    rng = _rng(seed)
    row = base_metric_row(simulation_id, scenario.scenario_id, "PVS_aware_area_laplace_approx", "reactive_area", df)

    y = df["y"].to_numpy(dtype=float)
    area = df["area"].to_numpy(dtype=float)
    lab_idx = df["lab_idx"].to_numpy(dtype=int)
    batch_idx = df["batch_idx"].to_numpy(dtype=int)
    n_labs = int(df["lab_idx"].nunique())
    n_batches = int(df["batch_idx"].nunique())

    def nlp(theta: np.ndarray) -> float:
        p = _unpack_hill_params(theta, n_labs, n_batches)
        mu = hill_response(area, p["y0"], p["ymax"], p["K_A"], p["h"])  # type: ignore[arg-type]
        mu = mu + p["lab_effect"][lab_idx] + p["batch_effect"][batch_idx]  # type: ignore[index]
        sigma = p["sigma0"] * (1.0 + p["rho_A"] * area)  # type: ignore[operator]
        if np.any(~np.isfinite(mu)) or np.any(sigma <= 0) or np.any(~np.isfinite(sigma)):
            return 1e30
        ll = student_t.logpdf(y, df=max(config.nu, 2.1), loc=mu, scale=sigma).sum()
        # Weak stabilizing priors/penalties on transformed parameters and effects.
        penalty = 0.0
        penalty += 0.5 * ((p["y0"] - 0.05) / 0.30) ** 2  # type: ignore[operator]
        penalty += 0.5 * ((theta[1] - np.log(0.9)) / 1.0) ** 2
        penalty += 0.5 * ((theta[2] - np.log(np.median(area))) / 1.0) ** 2
        penalty += 0.5 * ((theta[3] - np.log(3.0)) / 0.7) ** 2
        penalty += 0.5 * ((theta[4] - np.log(0.05)) / 1.0) ** 2
        penalty += 0.5 * ((theta[5] - np.log(max(config.rho_A, 0.05))) / 1.0) ** 2
        penalty += 0.5 * np.sum((p["lab_effect"] / 0.15) ** 2)  # type: ignore[operator]
        penalty += 0.5 * np.sum((p["batch_effect"] / 0.15) ** 2)  # type: ignore[operator]
        return float(-ll + penalty)

    x0 = np.zeros(6 + n_labs + n_batches)
    x0[0] = max(min(np.nanmin(y), 0.15), -0.05)
    x0[1] = np.log(max(np.nanmax(y) - np.nanmin(y), 0.5))
    x0[2] = np.log(np.median(area))
    x0[3] = np.log(max(config.h, 1.0))
    x0[4] = np.log(max(np.nanstd(y) * 0.5, 0.02))
    x0[5] = np.log(max(config.rho_A, 0.05))

    try:
        opt = minimize(nlp, x0, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-5})
        if not opt.success:
            warnings.warn(f"Laplace PVS optimization did not fully converge: {opt.message}", RuntimeWarning)
        theta_hat = opt.x
        cov = _safe_cov_from_hess_inv(opt.hess_inv, len(theta_hat))
        # Scale down over-dispersed BFGS covariance slightly for stability.
        cov = cov * 0.75
        theta_draws = rng.multivariate_normal(theta_hat, cov, size=n_draws, method="svd")

        mu_draws = np.zeros((n_draws, len(df)))
        pp = np.zeros_like(mu_draws)
        natural_samples: Dict[str, list] = {k: [] for k in ["y0", "ymax", "K_A", "h", "sigma0", "rho_A", "sigma_lab", "sigma_batch", "nu"]}
        for s in range(n_draws):
            p = _unpack_hill_params(theta_draws[s], n_labs, n_batches)
            mu = hill_response(area, p["y0"], p["ymax"], p["K_A"], p["h"])  # type: ignore[arg-type]
            mu = mu + p["lab_effect"][lab_idx] + p["batch_effect"][batch_idx]  # type: ignore[index]
            sigma = p["sigma0"] * (1.0 + p["rho_A"] * area)  # type: ignore[operator]
            sigma = np.maximum(sigma, 1e-6)
            mu_draws[s, :] = mu
            pp[s, :] = mu + sigma * rng.standard_t(df=max(config.nu, 2.1), size=len(df))
            for key in ["y0", "ymax", "K_A", "h", "sigma0", "rho_A", "sigma_lab", "sigma_batch"]:
                natural_samples[key].append(float(p[key]))  # type: ignore[arg-type]
            natural_samples["nu"].append(float(config.nu))

        samples_np = {k: np.asarray(v) for k, v in natural_samples.items()}
        theta_pvs, theta_ok = pvs_theta_from_samples(samples_np, config.y_phys_min, config.y_phys_max)
        pred_pvs = pvs_pred(pp, config.y_phys_min, config.y_phys_max)
        joint = pvs_joint(theta_ok, pp, config.y_phys_min, config.y_phys_max)
        y_hat = predictive_mean(pp)
        rd = residual_diagnostics(y, y_hat, design=df[["log_mass_z", "area"]].to_numpy())
        p_hat = _unpack_hill_params(theta_hat, n_labs, n_batches)
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
                "rhat_max": np.nan,
                "ess_bulk_min": np.nan,
                "ess_tail_min": np.nan,
                "n_divergences": np.nan,
                "bfmi_min": np.nan,
                "ppc_coverage": interval_coverage(y, pp, 0.95),
                "lab_variance_recovery": variance_recovery(float(p_hat["sigma_lab"]), config.sigma_lab_y),
                "batch_variance_recovery": variance_recovery(float(p_hat["sigma_batch"]), config.sigma_batch_y),
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
        row["diagnostic_flag"] = diagnostic_flag(row)  # type: ignore[arg-type]
        artifacts = {"posterior_predictive": pp, "mu_draws": mu_draws, "theta_draws_unconstrained": theta_draws, **samples_np}
        return row, artifacts
    except Exception as exc:
        row.update({"diagnostic_flag": f"fit_failed:{type(exc).__name__}", "runtime_seconds": time.perf_counter() - start})
        return row, {}
