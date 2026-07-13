"""Aggregate stress-test results as median [IQR] tables and manuscript JSON."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import json
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "rmse", "mae", "r2_or_pseudo_r2", "aic", "bic", "coverage_50", "coverage_80", "coverage_90", "coverage_95",
    "rhat_max", "ess_bulk_min", "ess_tail_min", "n_divergences", "bfmi_min", "ppc_coverage",
    "residual_heteroscedasticity_p", "residual_normality_p", "residual_tail_flag",
    "lab_variance_recovery", "batch_variance_recovery", "transition_region_error", "extrapolation_error",
    "pvs_theta", "pvs_pred", "pvs_descriptor", "pvs_joint", "prior_sensitivity_delta_pvs", "admissibility_sensitivity_delta_pvs",
    "runtime_seconds",
]


def median_iqr(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return "NA"
    q1 = vals.quantile(0.25)
    med = vals.quantile(0.50)
    q3 = vals.quantile(0.75)
    return f"{med:.4g} [{q1:.4g}, {q3:.4g}]"


def aggregate_results(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario_id", "model_name", "descriptor_used"]
    rows = []
    for keys, group in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(group))
        for col in NUMERIC_COLUMNS:
            if col in group.columns:
                row[col] = median_iqr(group[col])
        if "diagnostic_flag" in group.columns:
            row["diagnostic_flags"] = "; ".join(sorted(set(map(str, group["diagnostic_flag"].dropna()))))
        rows.append(row)
    return pd.DataFrame(rows)


def write_manuscript_numbers(metrics: pd.DataFrame, path: Path, config_dict: Dict) -> Dict:
    main = metrics.copy()
    out: Dict = {
        "config": config_dict,
        "n_metric_rows": int(len(metrics)),
        "scenarios": sorted(metrics["scenario_id"].dropna().unique().tolist()) if "scenario_id" in metrics else [],
        "models": sorted(metrics["model_name"].dropna().unique().tolist()) if "model_name" in metrics else [],
        "summary": {},
    }
    for (scenario, model), group in main.groupby(["scenario_id", "model_name"], dropna=False):
        key = f"{scenario}::{model}"
        out["summary"][key] = {}
        for col in NUMERIC_COLUMNS:
            if col in group.columns:
                out["summary"][key][col] = median_iqr(group[col])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def summarize_from_csv(metrics_csv: Path, output_dir: Path, config_dict: Dict | None = None) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_csv)
    summary = aggregate_results(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary_median_iqr.csv", index=False)
    write_manuscript_numbers(metrics, output_dir / "manuscript_numbers.json", config_dict or {})
    return summary
