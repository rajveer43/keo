"""Filesystem locations. Everything overridable by environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Repo root when running from a checkout (src/keo/paths.py -> ../../)."""
    return Path(__file__).resolve().parents[2]


def in_checkout() -> bool:
    """True when running from a source checkout, False when installed from a wheel."""
    return (project_root() / "pyproject.toml").is_file() and (project_root() / "configs").is_dir()


def configs_dir() -> Path:
    """Dataset and experiment YAMLs: the checkout's `configs/`, or the copy shipped inside the wheel."""
    env = os.environ.get("KEO_CONFIGS_DIR")
    if env:
        return Path(env)
    return project_root() / "configs" if in_checkout() else Path(__file__).resolve().parent / "_configs"


def data_dir() -> Path:
    """Downloaded datasets (large). Checkout: ./data. Installed: ~/.cache/keo/data. Each dataset gets its own directory."""
    env = os.environ.get("KEO_DATA_DIR")
    if env:
        return Path(env)
    return project_root() / "data" if in_checkout() else Path.home() / ".cache" / "keo" / "data"


def results_dir() -> Path:
    """Run outputs. Checkout: ./results. Installed: ./results under the current working directory."""
    env = os.environ.get("KEO_RESULTS_DIR")
    if env:
        return Path(env)
    return project_root() / "results" if in_checkout() else Path.cwd() / "results"
