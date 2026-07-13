"""Environment recording utilities."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata as metadata
import platform
import sys
from typing import Iterable


DEFAULT_PACKAGES = [
    "numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pymc", "arviz", "cmdstanpy",
]


def collect_environment(packages: Iterable[str] = DEFAULT_PACKAGES) -> str:
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
    ]
    for pkg in packages:
        try:
            version = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        lines.append(f"{pkg}: {version}")
    return "\n".join(lines) + "\n"


def save_environment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(collect_environment(), encoding="utf-8")
