"""Diagnostics and scoring utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def interval_coverage(y: np.ndarray, posterior_predictive: np.ndarray, level: float) -> float:
    """Coverage for posterior predictive draws shaped (n_draws, n_obs)."""
    y = np.asarray(y, dtype=float)
    pp = np.asarray(posterior_predictive, dtype=float)
    alpha = 1.0 - level
    lo = np.quantile(pp, alpha / 2.0, axis=0)
    hi = np.quantile(pp, 1.0 - alpha / 2.0, axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def predictive_mean(posterior_predictive: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(posterior_predictive, dtype=float), axis=0)


def residual_diagnostics(y: np.ndarray, y_hat: np.ndarray, design: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Return residual normality and heteroscedasticity diagnostics when statsmodels is available."""
    out: Dict[str, float] = {
        "residual_normality_p": float("nan"),
        "residual_heteroscedasticity_p": float("nan"),
        "residual_tail_flag": float("nan"),
    }
    residuals = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    try:
        from scipy.stats import jarque_bera, kurtosis

        jb = jarque_bera(residuals)
        out["residual_normality_p"] = float(jb.pvalue)
        out["residual_tail_flag"] = float(kurtosis(residuals, fisher=True, bias=False) > 1.0)
    except Exception:
        pass

    if design is not None:
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            import statsmodels.api as sm

            X = np.asarray(design, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = sm.add_constant(X, has_constant="add")
            bp = het_breuschpagan(residuals, X)
            out["residual_heteroscedasticity_p"] = float(bp[1])
        except Exception:
            pass
    return out


def transition_region_error(df: pd.DataFrame, y_hat: np.ndarray) -> float:
    mask = df["near_regime"].to_numpy(dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    return rmse(df.loc[mask, "true_mu"].to_numpy(), np.asarray(y_hat)[mask])


def extrapolation_error(true_mu: np.ndarray, pred_mu: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return rmse(np.asarray(true_mu)[mask], np.asarray(pred_mu)[mask])


def variance_recovery(estimated: Optional[float], truth: float) -> float:
    if estimated is None or not np.isfinite(estimated) or truth <= 0:
        return float("nan")
    return float(estimated / truth)


def summarize_arviz(idata) -> Dict[str, float]:
    """Extract MCMC diagnostics from an ArviZ InferenceData object."""
    out = {
        "rhat_max": float("nan"),
        "ess_bulk_min": float("nan"),
        "ess_tail_min": float("nan"),
        "n_divergences": float("nan"),
        "bfmi_min": float("nan"),
    }
    try:
        import arviz as az

        summary = az.summary(idata, kind="diagnostics")
        if "r_hat" in summary:
            out["rhat_max"] = float(np.nanmax(summary["r_hat"].to_numpy()))
        if "ess_bulk" in summary:
            out["ess_bulk_min"] = float(np.nanmin(summary["ess_bulk"].to_numpy()))
        if "ess_tail" in summary:
            out["ess_tail_min"] = float(np.nanmin(summary["ess_tail"].to_numpy()))
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            out["n_divergences"] = float(idata.sample_stats["diverging"].sum().item())
        try:
            bfmi = az.bfmi(idata)
            out["bfmi_min"] = float(np.nanmin(bfmi))
        except Exception:
            pass
    except Exception:
        pass
    return out


def diagnostic_flag(metrics: Dict[str, float]) -> str:
    flags = []
    rhat = metrics.get("rhat_max", float("nan"))
    if np.isfinite(rhat) and rhat > 1.01:
        flags.append("rhat_gt_1p01")
    div = metrics.get("n_divergences", float("nan"))
    if np.isfinite(div) and div > 0:
        flags.append("divergences")
    ess = metrics.get("ess_bulk_min", float("nan"))
    if np.isfinite(ess) and ess < 200:
        flags.append("low_ess")
    bp = metrics.get("residual_heteroscedasticity_p", float("nan"))
    if np.isfinite(bp) and bp < 0.05:
        flags.append("heteroscedastic_residuals")
    pvs = metrics.get("pvs_joint", float("nan"))
    if np.isfinite(pvs) and pvs < 0.8:
        flags.append("low_joint_pvs")
    return ";".join(flags) if flags else "ok"


def base_metric_row(
    simulation_id: int,
    scenario_id: str,
    model_name: str,
    descriptor_used: str,
    df: pd.DataFrame,
) -> Dict[str, object]:
    return {
        "simulation_id": simulation_id,
        "scenario_id": scenario_id,
        "model_name": model_name,
        "descriptor_used": descriptor_used,
        "n_obs": int(len(df)),
        "n_labs": int(df["lab_idx"].nunique()),
        "n_batches": int(df["batch_idx"].nunique()),
        "n_exposure_levels": int(df["exposure_idx"].nunique()),
        "n_replicates": int(df["replicate"].nunique()),
    }
