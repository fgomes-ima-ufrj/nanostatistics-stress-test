"""Frequentist OLS/ANOVA baselines for the synthetic stress test."""
from __future__ import annotations

from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd

from .config import DGPConfig
from .diagnostics import (
    base_metric_row,
    diagnostic_flag,
    mae,
    residual_diagnostics,
    r2_score,
    rmse,
    transition_region_error,
)


def fit_ols_anova(df: pd.DataFrame, config: DGPConfig, simulation_id: int, scenario_id: str) -> Tuple[Dict[str, object], object]:
    """Fit the conventional mass-based OLS/ANOVA baseline.

    The selected model includes fixed laboratory and batch indicators because this is
    the strongest conventional baseline among the three specified OLS variants.
    """
    start = time.perf_counter()
    try:
        import statsmodels.formula.api as smf

        model = smf.ols("y ~ log_mass_z + C(lab) + C(batch)", data=df).fit()
        y_hat = model.predict(df)
        design = df[["log_mass_z", "area"]].to_numpy()
        rd = residual_diagnostics(df["y"].to_numpy(), y_hat.to_numpy(), design=design)
        row: Dict[str, object] = base_metric_row(simulation_id, scenario_id, "OLS_ANOVA_mass_fixed_effects", "mass", df)
        row.update(
            {
                "rmse": rmse(df["y"].to_numpy(), y_hat.to_numpy()),
                "mae": mae(df["y"].to_numpy(), y_hat.to_numpy()),
                "r2_or_pseudo_r2": r2_score(df["y"].to_numpy(), y_hat.to_numpy()),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "coverage_50": np.nan,
                "coverage_80": np.nan,
                "coverage_90": np.nan,
                "coverage_95": np.nan,
                "rhat_max": np.nan,
                "ess_bulk_min": np.nan,
                "ess_tail_min": np.nan,
                "n_divergences": np.nan,
                "bfmi_min": np.nan,
                "ppc_coverage": np.nan,
                "lab_variance_recovery": np.nan,
                "batch_variance_recovery": np.nan,
                "transition_region_error": transition_region_error(df, y_hat.to_numpy()),
                "extrapolation_error": np.nan,
                "pvs_theta": np.nan,
                "pvs_pred": np.nan,
                "pvs_descriptor": np.nan,
                "pvs_joint": np.nan,
                "prior_sensitivity_delta_pvs": np.nan,
                "admissibility_sensitivity_delta_pvs": np.nan,
                "runtime_seconds": time.perf_counter() - start,
            }
        )
        row.update(rd)
        row["diagnostic_flag"] = diagnostic_flag(row)  # type: ignore[arg-type]
        return row, model
    except Exception as exc:
        row = base_metric_row(simulation_id, scenario_id, "OLS_ANOVA_mass_fixed_effects", "mass", df)
        row.update({"diagnostic_flag": f"fit_failed:{type(exc).__name__}", "runtime_seconds": time.perf_counter() - start})
        return row, None
