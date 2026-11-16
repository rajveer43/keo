"""Experiment configuration, grid expansion and cell identity.

A *cell* is one (dataset, normalize, n, k, algorithm spec, budget, seed) combination and has exactly one result row.
A *unit* is the set of cells of one (n, k, spec, seed) that share a fit invocation (reference Lloyd takes all its iteration
budgets from ONE run via snapshots; everything else is one fit per budget).
Cell ids are content hashes, so a re-run skips exactly the cells already stored (resume).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from keo.config import ConfigError, DatasetConfig
from keo.paths import configs_dir

ALGOS = ("lloyd", "emeans", "omeans")
BACKENDS = ("reference", "official")
DEFAULT_ADAPTER = {"lloyd": "faiss_lloyd_cpu", "emeans": "official_emeans_cpu", "omeans": "official_omeans"}
ALGO_ORDER = {a: i for i, a in enumerate(ALGOS)}
BACKEND_ORDER = {b: i for i, b in enumerate(BACKENDS)}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:n]


@dataclass(frozen=True)
class RunSpec:
    algo: str
    backend: str
    budgets: tuple[float, ...]
    params: dict[str, Any] = field(default_factory=dict)
    adapter: str | None = None  # official only

    @property
    def key(self) -> tuple[str, str, str | None, str]:
        """Identity of the algorithm configuration (budgets excluded) -- used to merge duplicate specs."""
        return (self.algo, self.backend, self.adapter, canonical_json(self.params))

    @property
    def budget_kind(self) -> str:
        return "iterations" if self.algo == "lloyd" else "epochs"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str
    normalize: bool | None  # None -> dataset default
    n_grid: tuple[int | None, ...]  # None = full train
    k_grid: tuple[int, ...]
    seeds: tuple[int, ...]
    runs: tuple[RunSpec, ...]
    baseline_iterations: int = 25
    train_eval_subset: int | None = None  # fixed-seed subset of train used for train MSE / dead / imbalance
    threads: int | None = None
    warmup: bool = True

    def fingerprint(self) -> str:
        return short_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "dataset": self.dataset, "normalize": self.normalize, "n": list(self.n_grid),
            "k": list(self.k_grid), "seeds": list(self.seeds), "baseline_iterations": self.baseline_iterations,
            "train_eval_subset": self.train_eval_subset, "threads": self.threads, "warmup": self.warmup,
            "runs": [{"algo": r.algo, "backend": r.backend, "budgets": list(r.budgets), "params": r.params,
                      "adapter": r.adapter} for r in self.runs],
        }


# ---------------------------------------------------------------------------------------------------- loading
def parse_experiment_config(raw: dict[str, Any], where: str = "<dict>") -> ExperimentConfig:
    def req(key: str) -> Any:
        if key not in raw:
            raise ConfigError(f"missing required key '{key}' in {where}")
        return raw[key]

    runs: list[RunSpec] = []
    for i, r in enumerate(req("runs")):
        algo, backend = r.get("algo"), r.get("backend", "reference")
        if algo not in ALGOS:
            raise ConfigError(f"{where}: runs[{i}].algo must be one of {ALGOS}, got {algo!r}")
        if backend not in BACKENDS:
            raise ConfigError(f"{where}: runs[{i}].backend must be one of {BACKENDS}, got {backend!r}")
        budgets = tuple(r.get("budgets") or ())
        if not budgets:
            raise ConfigError(f"{where}: runs[{i}] needs a non-empty 'budgets' list")
        if algo == "lloyd":
            if any(float(b) != int(b) or int(b) < 1 for b in budgets):
                raise ConfigError(f"{where}: lloyd budgets are iteration counts (positive integers)")
            budgets = tuple(int(b) for b in budgets)
        elif any(float(b) <= 0 for b in budgets):
            raise ConfigError(f"{where}: epoch budgets must be > 0")
        adapter = r.get("adapter") or (DEFAULT_ADAPTER[algo] if backend == "official" else None)
        if backend == "reference" and r.get("adapter"):
            raise ConfigError(f"{where}: runs[{i}]: 'adapter' only applies to backend: official")
        runs.append(RunSpec(algo, backend, tuple(float(b) if algo != "lloyd" else int(b) for b in budgets),
                            dict(r.get("params") or {}), adapter))
    n_grid = tuple(None if n is None else int(n) for n in req("n"))
    k_grid = tuple(int(k) for k in req("k"))
    seeds = tuple(int(s) for s in req("seeds"))
    if not (n_grid and k_grid and seeds):
        raise ConfigError(f"{where}: n, k and seeds must be non-empty")
    return ExperimentConfig(
        name=str(req("name")), dataset=str(req("dataset")),
        normalize=raw.get("normalize"), n_grid=n_grid, k_grid=k_grid, seeds=seeds, runs=tuple(runs),
        baseline_iterations=int(raw.get("baseline_iterations", 25)),
        train_eval_subset=raw.get("train_eval_subset"), threads=raw.get("threads"),
        warmup=bool(raw.get("warmup", True)),
    )


def load_experiment_config(name_or_path: str | Path) -> ExperimentConfig:
    p = Path(name_or_path)
    if p.suffix not in {".yaml", ".yml"}:
        p = configs_dir() / "experiments" / f"{name_or_path}.yaml"
    if not p.is_file():
        raise ConfigError(f"experiment config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: expected a mapping at top level")
    return parse_experiment_config(raw, where=str(p))


# ---------------------------------------------------------------------------------------------------- identity
def make_cell_id(dataset: str, normalize: bool, n: int, k: int, spec_key: tuple, budget: float, seed: int) -> str:
    return short_hash({"dataset": dataset, "normalize": normalize, "n": n, "k": k, "spec": list(spec_key),
                       "budget": float(budget), "seed": seed, "schema": 1})


BASELINE_REF_KEY = ("lloyd", "reference", None, canonical_json({}))
BASELINE_OFF_KEY = ("lloyd", "official", DEFAULT_ADAPTER["lloyd"], canonical_json({}))


@dataclass
class Unit:
    n: int
    k: int
    seed: int
    spec: RunSpec
    budgets: list[float]
    baseline: bool = False  # True if this unit exists only to provide the Lloyd@25 reference


@dataclass
class Plan:
    units: list[Unit]
    skipped: list[dict[str, Any]]

    def n_cells(self) -> int:
        return sum(len(u.budgets) for u in self.units)


def expand(exp: ExperimentConfig, ds: DatasetConfig, availability: dict[str, tuple[bool, str]] | None = None) -> Plan:
    """Expand the grid into units. `availability` maps official adapter key -> (available, reason)."""
    availability = availability or {}
    # merge specs with identical algorithm configuration, and make sure the Lloyd@25 reference exists
    merged: dict[tuple, RunSpec] = {}
    for r in exp.runs:
        if r.key in merged:
            budgets = tuple(sorted(set(merged[r.key].budgets) | set(r.budgets)))
            merged[r.key] = RunSpec(r.algo, r.backend, budgets, r.params, r.adapter)
        else:
            merged[r.key] = RunSpec(r.algo, r.backend, tuple(sorted(set(r.budgets))), r.params, r.adapter)
    baseline_added = False
    if BASELINE_REF_KEY not in merged:
        merged[BASELINE_REF_KEY] = RunSpec("lloyd", "reference", (exp.baseline_iterations,), {}, None)
        baseline_added = True
    elif exp.baseline_iterations not in merged[BASELINE_REF_KEY].budgets:
        old = merged[BASELINE_REF_KEY]
        merged[BASELINE_REF_KEY] = RunSpec("lloyd", "reference", tuple(sorted(set(old.budgets) | {exp.baseline_iterations})), {}, None)
        baseline_added = True

    units: list[Unit] = []
    skipped: list[dict[str, Any]] = []
    seen_skip: set[tuple] = set()

    def skip(reason: str, **info: Any) -> None:
        marker = (reason, tuple(sorted((k, str(v)) for k, v in info.items())))
        if marker not in seen_skip:
            seen_skip.add(marker)
            skipped.append({"reason": reason, **info})

    for n_cfg in exp.n_grid:
        n = ds.n_train if n_cfg is None else min(int(n_cfg), ds.n_train)
        for k in exp.k_grid:
            if k > n or n / k < ds.min_points_per_cluster:
                skip(f"n/k < {ds.min_points_per_cluster} (or k > n)", n=n, k=k)
                continue
            for spec in merged.values():
                if spec.backend == "official":
                    ok, why = availability.get(spec.adapter or "", (False, "adapter status unknown"))
                    if not ok:
                        skip(f"adapter unavailable: {why}", algo=spec.algo, adapter=spec.adapter)
                        continue
                for seed in exp.seeds:
                    units.append(Unit(n, k, seed, spec, list(spec.budgets),
                                      baseline=(spec.key == BASELINE_REF_KEY and baseline_added)))
    # within each (n, k, seed): reference Lloyd (default params) first, so the Lloyd@25 reference exists before anything is compared to it
    units.sort(key=lambda u: (u.n, u.k, u.seed, ALGO_ORDER[u.spec.algo], BACKEND_ORDER[u.spec.backend],
                              0 if u.spec.key in (BASELINE_REF_KEY, BASELINE_OFF_KEY) else 1, u.spec.key[2] or "", u.spec.key[3]))
    return Plan(units, skipped)
