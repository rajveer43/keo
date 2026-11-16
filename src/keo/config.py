"""Dataset configuration loader (YAML -> validated dataclasses).

Adding a dataset = adding a YAML file under configs/datasets/ (no code change).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from keo.paths import configs_dir


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SourceConfig:
    type: str
    url: str
    filename: str
    format: str
    train_key: str
    eval_key: str
    sha256: str = ""
    expected_bytes: int | None = None


@dataclass(frozen=True)
class SplitConfig:
    val_fraction: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    description: str
    source: SourceConfig
    d: int
    n_train: int
    n_eval: int
    normalize: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)
    k_grid: tuple[int, ...] = ()
    n_subset_grid: tuple[int | None, ...] = ()
    min_points_per_cluster: int = 8


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return d[key]


def parse_dataset_config(raw: dict[str, Any], where: str = "<dict>") -> DatasetConfig:
    src_raw = _require(raw, "source", where)
    if src_raw.get("type") != "url":
        raise ConfigError(f"{where}: only source.type == 'url' is supported, got {src_raw.get('type')!r}")
    if src_raw.get("format") != "hdf5":
        raise ConfigError(f"{where}: only source.format == 'hdf5' is supported, got {src_raw.get('format')!r}")
    source = SourceConfig(
        type=src_raw["type"],
        url=_require(src_raw, "url", where),
        filename=_require(src_raw, "filename", where),
        format=src_raw["format"],
        train_key=_require(src_raw, "train_key", where),
        eval_key=_require(src_raw, "eval_key", where),
        sha256=str(src_raw.get("sha256") or ""),
        expected_bytes=src_raw.get("expected_bytes"),
    )
    split_raw = raw.get("split", {}) or {}
    split = SplitConfig(
        val_fraction=float(split_raw.get("val_fraction", 0.5)),
        seed=int(split_raw.get("seed", 0)),
    )
    if not 0.0 < split.val_fraction < 1.0:
        raise ConfigError(f"{where}: split.val_fraction must be in (0,1)")
    return DatasetConfig(
        name=_require(raw, "name", where),
        description=str(raw.get("description", "")),
        source=source,
        d=int(_require(raw, "d", where)),
        n_train=int(_require(raw, "n_train", where)),
        n_eval=int(_require(raw, "n_eval", where)),
        normalize=bool(raw.get("normalize", True)),
        split=split,
        k_grid=tuple(int(k) for k in raw.get("k_grid", [])),
        n_subset_grid=tuple(None if n is None else int(n) for n in raw.get("n_subset_grid", [])),
        min_points_per_cluster=int(raw.get("min_points_per_cluster", 8)),
    )


def load_dataset_config(name_or_path: str | Path) -> DatasetConfig:
    """Load by registry name (configs/datasets/<name>.yaml) or by explicit YAML path."""
    p = Path(name_or_path)
    if p.suffix not in {".yaml", ".yml"}:
        p = configs_dir() / "datasets" / f"{name_or_path}.yaml"
    if not p.is_file():
        raise ConfigError(f"dataset config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: expected a mapping at top level")
    return parse_dataset_config(raw, where=str(p))


def list_dataset_configs() -> list[str]:
    d = configs_dir() / "datasets"
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []
