"""Monte Carlo runner for the Nanostatistics synthetic stress test."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import traceback

import numpy as np
import pandas as pd

from .config import (
    DGPConfig,
    InferenceConfig,
    OutputConfig,
    RunConfig,
    config_to_dict,
    default_scenarios,
    ensure_output_dirs,
    save_config_json,
    scenario_to_dgp,
)
from .dgp import simulate_dataset
from .environment import save_environment
from .models_frequentist import fit_ols_anova
from .models_laplace import fit_mass_laplace, fit_pvs_laplace
from .summarize_results import aggregate_results, write_manuscript_numbers
from .plots import plot_dgp, plot_metrics, plot_pvs


def _fit_bayesian_models(df: pd.DataFrame, dgp: DGPConfig, inference: InferenceConfig, scenario, simulation_id: int, seed: int, paths: Dict[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    engine = inference.engine
    if engine == "laplace_fast":
        row, _ = fit_mass_laplace(df, dgp, scenario, simulation_id, seed + 1000, n_draws=inference.n_laplace_draws)
        rows.append(row)
        row, _ = fit_pvs_laplace(df, dgp, scenario, simulation_id, seed + 2000, n_draws=inference.n_laplace_draws, maxiter=inference.max_laplace_iter)
        rows.append(row)
        return rows

    if engine == "pymc":
        from .models_bayesian_mass import fit_mass_pymc
        from .models_bayesian_pvs import fit_pvs_pymc

        trace_mass = paths["traces"] / f"trace_mass_s{scenario.scenario_id}_sim{simulation_id}.nc"
        trace_pvs = paths["traces"] / f"trace_pvs_s{scenario.scenario_id}_sim{simulation_id}.nc"
        row, _ = fit_mass_pymc(df, dgp, inference, scenario, simulation_id, seed + 1000, trace_mass)
        rows.append(row)
        row, _ = fit_pvs_pymc(df, dgp, inference, scenario, simulation_id, seed + 2000, trace_pvs)
        rows.append(row)
        return rows

    if engine == "stan":
        from .models_stan import fit_mass_stan, fit_pvs_stan

        row, _ = fit_mass_stan(df, dgp, inference, scenario, simulation_id, seed + 1000, paths["stan_models"])
        rows.append(row)
        row, _ = fit_pvs_stan(df, dgp, inference, scenario, simulation_id, seed + 2000, paths["stan_models"])
        rows.append(row)
        return rows

    raise ValueError(f"Unknown engine: {engine}")


def run_stress_test(config: RunConfig) -> pd.DataFrame:
    paths = ensure_output_dirs(config.output)
    save_config_json(config, paths["root"] / "config_used.json")
    save_environment(paths["root"] / "environment.txt")

    metrics_rows: List[Dict[str, object]] = []
    first_df: Optional[pd.DataFrame] = None

    for scenario in config.scenarios:
        dgp = scenario_to_dgp(config.dgp, scenario)
        for sim in range(config.inference.n_sim):
            seed = config.inference.random_seed + 100_000 * sim + abs(hash(scenario.scenario_id)) % 10_000
            try:
                df, truth = simulate_dataset(dgp, seed=seed, simulation_id=sim, scenario_id=scenario.scenario_id)
                if first_df is None:
                    first_df = df.copy()
                df_path = paths["simulated_data"] / f"data_{scenario.scenario_id}_sim{sim}.csv"
                truth_path = paths["simulated_data"] / f"truth_{scenario.scenario_id}_sim{sim}.json"
                df.to_csv(df_path, index=False)
                truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

                row, _ = fit_ols_anova(df, dgp, sim, scenario.scenario_id)
                metrics_rows.append(row)
                metrics_rows.extend(_fit_bayesian_models(df, dgp, config.inference, scenario, sim, seed, paths))
            except Exception as exc:
                metrics_rows.append(
                    {
                        "simulation_id": sim,
                        "scenario_id": scenario.scenario_id,
                        "model_name": "RUNNER_FAILURE",
                        "diagnostic_flag": f"runner_failed:{type(exc).__name__}",
                        "traceback": traceback.format_exc(),
                    }
                )

            # Incremental save after every simulation to preserve partial runs.
            metrics = pd.DataFrame(metrics_rows)
            metrics.to_csv(paths["tables"] / "metrics_raw.csv", index=False)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(paths["tables"] / "metrics_raw.csv", index=False)
    summary = aggregate_results(metrics)
    summary.to_csv(paths["tables"] / "summary_median_iqr.csv", index=False)
    write_manuscript_numbers(metrics, paths["tables"] / "manuscript_numbers.json", config_to_dict(config))

    if first_df is not None:
        plot_dgp(first_df, paths["figures"] / "figure_S1_dgp.png")
    plot_metrics(metrics, paths["figures"] / "figure_S2_model_comparison.png")
    plot_pvs(metrics, paths["figures"] / "figure_S3_pvs.png")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nanostatistics synthetic stress test.")
    parser.add_argument("--engine", choices=["laplace_fast", "pymc", "stan"], default="laplace_fast")
    parser.add_argument("--n-sim", type=int, default=10)
    parser.add_argument("--include-sweeps", action="store_true")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--no-progressbar", action="store_true")
    parser.add_argument("--save-traces", action="store_true", default=True)
    parser.add_argument("--no-save-traces", dest="save_traces", action="store_false")
    parser.add_argument("--n-laplace-draws", type=int, default=2000)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RunConfig:
    inference = InferenceConfig(
        engine=args.engine,
        n_sim=args.n_sim,
        random_seed=args.seed,
        chains=args.chains,
        tune=args.tune,
        draws=args.draws,
        target_accept=args.target_accept,
        progressbar=not args.no_progressbar,
        save_traces=args.save_traces,
        n_laplace_draws=args.n_laplace_draws,
    )
    output = OutputConfig(root=Path(args.output))
    return RunConfig(inference=inference, output=output, scenarios=default_scenarios(include_sweeps=args.include_sweeps))


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run_stress_test(config)


if __name__ == "__main__":
    main()
