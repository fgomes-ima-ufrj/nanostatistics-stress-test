"""Configuration objects for the Nanostatistics synthetic stress test.

The defaults are intentionally conservative for local development. For manuscript-scale
runs, use the CLI flags shown in README.md, for example --n-sim 100 --draws 4000
--tune 2000 --target-accept 0.95.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import json


PriorMode = Literal["nominal", "narrow", "broad", "mildly_misspecified"]
YPhysMode = Literal["nominal", "narrow", "broad"]
Engine = Literal["pymc", "stan", "laplace_fast"]


@dataclass(frozen=True)
class DGPConfig:
    """Data-generating process parameters.

    Indices follow the manuscript notation:
    i = laboratory, b = batch, k = exposure level, r = replicate.
    """

    n_labs: int = 3
    n_batches: int = 4
    n_exposure_levels: int = 8
    n_replicates: int = 3

    mass_min: float = 0.05
    mass_max: float = 5.0
    mass_spacing: Literal["log", "linear"] = "log"
    epsilon: float = 1e-9

    # log(A) = alpha_A + beta_A log(m + eps) + lab + batch + error
    alpha_A: float = -0.25
    beta_A: float = 0.95
    sigma_lab_A: float = 0.18
    sigma_batch_A: float = 0.12
    sigma_obs_A: float = 0.10

    # mu = y0 + ymax * A^h / (K_A^h + A^h) + lab + batch
    y0: float = 0.04
    ymax: float = 0.90
    K_A: float = 0.95
    h: float = 3.0
    sigma_lab_y: float = 0.025
    sigma_batch_y: float = 0.020

    # y ~ StudentT(nu, mu, sigma0 * (1 + rho_A * A))
    nu: float = 5.0
    sigma0: float = 0.035
    rho_A: float = 0.10

    # Physical predictive domain for the normalized response
    y_phys_min: float = 0.0
    y_phys_max: float = 1.0


@dataclass(frozen=True)
class InferenceConfig:
    """Inference settings for Bayesian engines and fast approximations."""

    engine: Engine = "laplace_fast"
    chains: int = 4
    tune: int = 1000
    draws: int = 1000
    target_accept: float = 0.90
    random_seed: int = 20260630
    n_sim: int = 10
    n_laplace_draws: int = 2000
    max_laplace_iter: int = 2000
    save_traces: bool = True
    progressbar: bool = True

    # For PyMC only. Use a fixed seed per simulation/chain to improve reproducibility.
    cores: Optional[int] = None


@dataclass(frozen=True)
class ScenarioConfig:
    """One stress-test scenario."""

    scenario_id: str = "descriptor_mismatch_nominal"
    description: str = "Main descriptor-mismatch scenario: response governed by reactive area while mass is the conventional descriptor."
    n_replicates: int = 3
    nu: float = 5.0
    rho_A: float = 0.10
    h: float = 3.0
    K_A: float = 0.95
    prior_mode: PriorMode = "nominal"
    y_phys_mode: YPhysMode = "nominal"


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("outputs")
    simulated_data_dir: str = "simulated_data"
    traces_dir: str = "posterior_traces"
    summaries_dir: str = "model_summaries"
    diagnostics_dir: str = "diagnostics"
    tables_dir: str = "tables"
    figures_dir: str = "figures"
    stan_models_dir: str = "stan_models"


@dataclass(frozen=True)
class RunConfig:
    dgp: DGPConfig = field(default_factory=DGPConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scenarios: Tuple[ScenarioConfig, ...] = field(default_factory=lambda: (ScenarioConfig(),))


def scenario_to_dgp(base: DGPConfig, scenario: ScenarioConfig) -> DGPConfig:
    """Apply a scenario to the DGP configuration."""
    y_min, y_max = physical_y_bounds(scenario.y_phys_mode)
    return replace(
        base,
        n_replicates=scenario.n_replicates,
        nu=scenario.nu,
        rho_A=scenario.rho_A,
        h=scenario.h,
        K_A=scenario.K_A,
        y_phys_min=y_min,
        y_phys_max=y_max,
    )


def physical_y_bounds(mode: YPhysMode) -> Tuple[float, float]:
    """Scenario-specific predictive admissibility domain."""
    if mode == "narrow":
        return 0.02, 0.98
    if mode == "broad":
        return -0.05, 1.05
    return 0.0, 1.0


def default_scenarios(include_sweeps: bool = False) -> Tuple[ScenarioConfig, ...]:
    """Return the scenario matrix.

    include_sweeps=False keeps the main manuscript scenario only.
    include_sweeps=True adds sparsity, heavy-tail, heteroscedasticity,
    regime-sensitivity, and prior/admissibility sensitivity scenarios.
    """
    scenarios: List[ScenarioConfig] = [ScenarioConfig()]
    if not include_sweeps:
        return tuple(scenarios)

    # Sparsity sweep
    for r in (1, 2, 3, 5):
        scenarios.append(
            ScenarioConfig(
                scenario_id=f"sparsity_replicates_{r}",
                description=f"Sparsity sweep with {r} replicate(s) per lab-batch-exposure cell.",
                n_replicates=r,
            )
        )

    # Heavy-tail sweep
    for nu in (3.0, 5.0, 10.0, 30.0):
        scenarios.append(
            ScenarioConfig(
                scenario_id=f"heavy_tail_nu_{int(nu)}",
                description=f"Heavy-tail sensitivity with Student-t nu={nu}.",
                nu=nu,
            )
        )

    # Heteroscedasticity sweep
    for rho in (0.0, 0.25, 0.50, 1.00):
        scenarios.append(
            ScenarioConfig(
                scenario_id=f"heteroscedasticity_rho_{str(rho).replace('.', 'p')}",
                description=f"Heteroscedasticity sensitivity with rho_A={rho}.",
                rho_A=rho,
            )
        )

    # Regime-sensitivity sweep
    for h in (1.0, 2.0, 4.0, 8.0):
        scenarios.append(
            ScenarioConfig(
                scenario_id=f"regime_h_{str(h).replace('.', 'p')}",
                description=f"Regime-sensitivity sweep with Hill coefficient h={h}.",
                h=h,
            )
        )
    for label, K in (("low", 0.45), ("medium", 0.95), ("high", 1.80)):
        scenarios.append(
            ScenarioConfig(
                scenario_id=f"regime_KA_{label}",
                description=f"Regime-location sensitivity with K_A={K}.",
                K_A=K,
            )
        )

    # Prior/admissibility sensitivity
    for prior in ("narrow", "nominal", "broad", "mildly_misspecified"):
        scenarios.append(
            ScenarioConfig(
                scenario_id=f"prior_{prior}",
                description=f"Prior sensitivity scenario: {prior}.",
                prior_mode=prior,  # type: ignore[arg-type]
            )
        )
    for ymode in ("narrow", "nominal", "broad"):
        scenarios.append(
            ScenarioConfig(
                scenario_id=f"yphys_{ymode}",
                description=f"Predictive admissibility-domain sensitivity: {ymode}.",
                y_phys_mode=ymode,  # type: ignore[arg-type]
            )
        )

    # Remove duplicate IDs while preserving order.
    seen = set()
    unique: List[ScenarioConfig] = []
    for s in scenarios:
        if s.scenario_id not in seen:
            unique.append(s)
            seen.add(s.scenario_id)
    return tuple(unique)


def config_to_dict(config: RunConfig) -> Dict[str, Any]:
    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return [convert(x) for x in obj]
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(asdict(config))


def save_config_json(config: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_output_dirs(output: OutputConfig) -> Dict[str, Path]:
    root = output.root
    paths = {
        "root": root,
        "simulated_data": root / output.simulated_data_dir,
        "traces": root / output.traces_dir,
        "summaries": root / output.summaries_dir,
        "diagnostics": root / output.diagnostics_dir,
        "tables": root / output.tables_dir,
        "figures": root / output.figures_dir,
        "stan_models": root / output.stan_models_dir,
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
