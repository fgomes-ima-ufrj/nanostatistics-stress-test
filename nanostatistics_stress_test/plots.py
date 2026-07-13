"""Publication-oriented diagnostic plots for the stress test."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def plot_dgp(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.scatter(df["mass"], df["area"], alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nominal mass")
    ax.set_ylabel("Reactive area")
    ax.set_title("Descriptor mismatch")

    ax = axes[0, 1]
    ax.scatter(df["area"], df["y"], alpha=0.5, label="observed")
    order = np.argsort(df["area"].to_numpy())
    ax.plot(df["area"].to_numpy()[order], df["true_mu"].to_numpy()[order], linewidth=1.5, label="true mean")
    ax.set_xlabel("Reactive area")
    ax.set_ylabel("Response")
    ax.set_title("Hill-type response")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(df["area"], df["true_sigma"], alpha=0.7)
    ax.set_xlabel("Reactive area")
    ax.set_ylabel("True sigma")
    ax.set_title("Heteroscedastic noise")

    ax = axes[1, 1]
    df.boxplot(column="y", by="lab", ax=ax)
    ax.set_title("Laboratory-level structure")
    ax.set_xlabel("Laboratory")
    ax.set_ylabel("Response")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = metrics.pivot_table(index="model_name", values="rmse", aggfunc="median")
    pivot.plot(kind="bar", ax=ax, legend=False)
    ax.set_ylabel("Median RMSE")
    ax.set_xlabel("Model")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pvs(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pvs_cols = [c for c in ["pvs_theta", "pvs_pred", "pvs_joint"] if c in metrics.columns]
    data = metrics.dropna(subset=pvs_cols, how="all")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    labels = []
    for model, group in data.groupby("model_name"):
        vals = pd.to_numeric(group["pvs_joint"], errors="coerce").dropna()
        if len(vals):
            rows.append(vals.to_numpy())
            labels.append(model)
    if rows:
        ax.boxplot(rows, labels=labels, vert=True)
        ax.set_ylabel("Joint PVS")
        ax.set_title("Physical Validity Score diagnostics")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
