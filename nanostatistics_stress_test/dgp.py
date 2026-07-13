"""Synthetic hierarchical data-generating process for Nanostatistics stress tests."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .config import DGPConfig


def mass_levels(config: DGPConfig) -> np.ndarray:
    if config.mass_spacing == "linear":
        return np.linspace(config.mass_min, config.mass_max, config.n_exposure_levels)
    return np.geomspace(config.mass_min, config.mass_max, config.n_exposure_levels)


def hill_response(area: np.ndarray, y0: float, ymax: float, K_A: float, h: float) -> np.ndarray:
    area = np.asarray(area, dtype=float)
    area_pos = np.maximum(area, 1e-12)
    numerator = np.power(area_pos, h)
    denominator = np.power(K_A, h) + numerator
    return y0 + ymax * numerator / denominator


def simulate_dataset(config: DGPConfig, seed: int, simulation_id: int = 0, scenario_id: str = "main") -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Simulate one complete lab-batch-exposure-replicate dataset.

    Returns
    -------
    df
        Long-format dataset with true latent quantities retained for diagnostics.
    truth
        Dictionary containing DGP parameters and generated random effects.
    """
    rng = np.random.default_rng(seed)
    masses = mass_levels(config)

    lab_A = rng.normal(0.0, config.sigma_lab_A, size=config.n_labs)
    batch_A = rng.normal(0.0, config.sigma_batch_A, size=config.n_batches)
    lab_y = rng.normal(0.0, config.sigma_lab_y, size=config.n_labs)
    batch_y = rng.normal(0.0, config.sigma_batch_y, size=config.n_batches)

    rows = []
    obs_id = 0
    for i in range(config.n_labs):
        for b in range(config.n_batches):
            for k, m in enumerate(masses):
                for r in range(config.n_replicates):
                    e_A = rng.normal(0.0, config.sigma_obs_A)
                    log_A = config.alpha_A + config.beta_A * np.log(m + config.epsilon) + lab_A[i] + batch_A[b] + e_A
                    A = float(np.exp(log_A))
                    true_mu_no_effect = float(hill_response(A, config.y0, config.ymax, config.K_A, config.h))
                    mu = true_mu_no_effect + lab_y[i] + batch_y[b]
                    sigma = config.sigma0 * (1.0 + config.rho_A * A)
                    y = mu + sigma * rng.standard_t(df=config.nu)
                    rows.append(
                        {
                            "simulation_id": simulation_id,
                            "scenario_id": scenario_id,
                            "obs_id": obs_id,
                            "lab": f"L{i + 1}",
                            "batch": f"B{b + 1}",
                            "lab_idx": i,
                            "batch_idx": b,
                            "exposure_idx": k,
                            "replicate": r,
                            "mass": float(m),
                            "log_mass": float(np.log(m + config.epsilon)),
                            "area": A,
                            "log_area": float(log_A),
                            "true_mu_no_effect": true_mu_no_effect,
                            "true_mu": float(mu),
                            "true_sigma": float(sigma),
                            "y": float(y),
                            "near_regime": bool(abs(A - config.K_A) <= 0.25 * config.K_A),
                        }
                    )
                    obs_id += 1

    df = pd.DataFrame(rows)
    df["log_mass_z"] = (df["log_mass"] - df["log_mass"].mean()) / df["log_mass"].std(ddof=0)
    df["log_area_z"] = (df["log_area"] - df["log_area"].mean()) / df["log_area"].std(ddof=0)

    truth: Dict[str, object] = {
        "config": asdict(config),
        "mass_levels": masses.tolist(),
        "lab_effect_A": lab_A.tolist(),
        "batch_effect_A": batch_A.tolist(),
        "lab_effect_y": lab_y.tolist(),
        "batch_effect_y": batch_y.tolist(),
        "seed": seed,
        "simulation_id": simulation_id,
        "scenario_id": scenario_id,
    }
    return df, truth


def extrapolation_grid(config: DGPConfig, n_points: int = 60, factor: float = 1.5) -> pd.DataFrame:
    """Create a deterministic extrapolation grid beyond the observed mass range.

    Lab and batch are fixed to the first levels; random effects are set to zero by using
    the expected log-area transformation only.
    """
    min_m = config.mass_min
    max_m = config.mass_max * factor
    masses = np.geomspace(min_m, max_m, n_points)
    log_A = config.alpha_A + config.beta_A * np.log(masses + config.epsilon)
    area = np.exp(log_A)
    true_mu = hill_response(area, config.y0, config.ymax, config.K_A, config.h)
    grid = pd.DataFrame(
        {
            "mass": masses,
            "log_mass": np.log(masses + config.epsilon),
            "area": area,
            "log_area": log_A,
            "lab": "L1",
            "batch": "B1",
            "lab_idx": 0,
            "batch_idx": 0,
            "true_mu": true_mu,
            "near_regime": np.abs(area - config.K_A) <= 0.25 * config.K_A,
            "is_extrapolation": masses > config.mass_max,
        }
    )
    grid["log_mass_z"] = (grid["log_mass"] - grid["log_mass"].mean()) / grid["log_mass"].std(ddof=0)
    grid["log_area_z"] = (grid["log_area"] - grid["log_area"].mean()) / grid["log_area"].std(ddof=0)
    return grid
