"""Physical Validity Score utilities."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple
import numpy as np
import pandas as pd


def pvs_descriptor(area: np.ndarray, lower: float = 0.0, upper: float = np.inf) -> float:
    area = np.asarray(area, dtype=float)
    return float(np.mean((area > lower) & (area < upper)))


def pvs_pred(posterior_predictive: np.ndarray, y_min: float, y_max: float) -> float:
    pp = np.asarray(posterior_predictive, dtype=float)
    return float(np.mean((pp >= y_min) & (pp <= y_max)))


def pvs_theta_from_samples(samples: Mapping[str, np.ndarray], y_min: float = 0.0, y_max: float = 1.0) -> Tuple[float, np.ndarray]:
    """Calculate parameter-space PVS for natural-scale posterior samples.

    Expected keys for the PVS-aware Hill model:
    y0, ymax, K_A, h, sigma0, rho_A, sigma_lab, sigma_batch, nu.

    For models without a mechanistic parameterization, return NaN and an empty mask.
    """
    required = ["y0", "ymax", "K_A", "h", "sigma0", "rho_A", "sigma_lab", "sigma_batch", "nu"]
    if not all(k in samples for k in required):
        return float("nan"), np.array([], dtype=bool)

    arrays = {k: np.asarray(samples[k], dtype=float).reshape(-1) for k in required}
    n = len(arrays[required[0]])
    ok = np.ones(n, dtype=bool)
    ok &= arrays["y0"] >= y_min
    ok &= arrays["ymax"] >= 0
    ok &= arrays["K_A"] > 0
    ok &= arrays["h"] > 0
    ok &= arrays["sigma0"] > 0
    ok &= arrays["rho_A"] >= 0
    ok &= arrays["sigma_lab"] > 0
    ok &= arrays["sigma_batch"] > 0
    ok &= arrays["nu"] > 2
    # Parameter-space response bound. This makes PVS_theta informative even when
    # positivity is imposed in priors/sampling.
    ok &= (arrays["y0"] + arrays["ymax"]) <= y_max
    return float(np.mean(ok)), ok


def pvs_joint(theta_ok: Optional[np.ndarray], posterior_predictive: np.ndarray, y_min: float, y_max: float) -> float:
    pp = np.asarray(posterior_predictive, dtype=float)
    pred_ok = (pp >= y_min) & (pp <= y_max)
    if theta_ok is None or len(theta_ok) == 0:
        return float(np.mean(pred_ok))
    theta_ok = np.asarray(theta_ok, dtype=bool).reshape(-1)
    n_draws = min(len(theta_ok), pred_ok.shape[0])
    return float(np.mean(theta_ok[:n_draws, None] & pred_ok[:n_draws, :]))


def flatten_pymc_posterior(idata, var_names) -> Dict[str, np.ndarray]:
    """Flatten chain/draw posterior variables from ArviZ InferenceData."""
    out: Dict[str, np.ndarray] = {}
    posterior = idata.posterior
    for name in var_names:
        if name in posterior:
            arr = posterior[name].values
            out[name] = arr.reshape(-1, *arr.shape[2:])
    return out


def predictive_from_idata(idata, observed_name: str = "y_obs") -> np.ndarray:
    """Return posterior predictive draws with shape (n_draws, n_obs)."""
    pp = idata.posterior_predictive[observed_name].values
    return pp.reshape(-1, pp.shape[-1])
