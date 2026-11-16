"""Experiment runner: grid expansion -> resumable execution -> one validated parquet row per cell.

Design points (docs/SPEC.md §7, ASSUMPTIONS A28-A29, A53):
  * one shared init per (n, k, seed): random rows, identical for K, E and O (and passed to official adapters that accept it);
  * all quality numbers use EXACT assignment and float64 distances; relative MSE is vs the reference Lloyd@25 of the SAME
    seed (and vs the official Lloyd@25 when that row exists);
  * reference Lloyd takes all its iteration budgets from ONE run (snapshots), everything else is one fit per budget;
  * a warm-up fit (discarded) precedes the first timed fit of every (n, k, spec) group on the reference backend;
  * failures become `status="failed"` rows (never silent); unavailable adapters / invalid (n, k) are listed, not run.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keo import __version__
from keo.algos import Budget, EMeans, Lloyd, OMeans
from keo.algos.adapters._worker_client import AdapterUnavailable
from keo.algos.adapters.registry import adapter_status, get_adapter
from keo.algos.init import random_row_init
from keo.bench.experiment import (
    BASELINE_OFF_KEY, BASELINE_REF_KEY, ExperimentConfig, RunSpec, Unit, canonical_json, expand, make_cell_id,
)
from keo.bench.provenance import git_sha, peak_rss_mb, utc_now
from keo.bench.schema import new_row
from keo.bench.store import ResultStore
from keo.config import ConfigError, DatasetConfig
from keo.data.prep import Prepared, prepare, subset_train
from keo.metrics import assign, cluster_stats, mse
from keo.runtime import configure_threads, hardware_descriptor

REFERENCE = {"lloyd": Lloyd, "emeans": EMeans, "omeans": OMeans}


@dataclass
class RunSummary:
    out_dir: Path
    n_cells_planned: int = 0
    n_units: int = 0
    n_cells_resumed: int = 0  # already stored, skipped
    n_ok: int = 0
    n_failed: int = 0
    seconds: float = 0.0
    skipped: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False
    results_path: Path | None = None

    def line(self) -> str:
        tag = "DRY RUN " if self.dry_run else ""
        return (f"{tag}cells planned={self.n_cells_planned} (units={self.n_units}) resumed={self.n_cells_resumed} "
                f"ok={self.n_ok} failed={self.n_failed} skipped-groups={len(self.skipped)} in {self.seconds:.1f}s")


def make_algo(spec: RunSpec):
    if spec.backend == "reference":
        return REFERENCE[spec.algo](**spec.params)
    return get_adapter(spec.adapter or "", **spec.params)


def _py(v: Any) -> Any:
    """numpy -> python scalars; NaN/inf -> None (parquet nulls)."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _rel(x: float | None, ref: float | None) -> float | None:
    return None if (x is None or ref is None or ref == 0) else float((x - ref) / ref)


class _Ctx:
    """Everything that is constant for one run_experiment call."""

    def __init__(self, exp: ExperimentConfig, ds: DatasetConfig, prep: Prepared, store: ResultStore, hw: dict[str, Any]):
        self.exp, self.ds, self.prep, self.store = exp, ds, prep, store
        self.hw_json = canonical_json(hw)
        self.torch_threads = int(hw.get("torch_threads") or 0)
        self.git = git_sha()
        self.cfg_hash = exp.fingerprint()
        self._train: dict[int, np.ndarray] = {}
        self._eval: dict[int, np.ndarray] = {}
        self._init: dict[tuple[int, int, int], np.ndarray] = {}
        self.warmed: set[tuple] = set()

    def train(self, n: int) -> np.ndarray:
        if n not in self._train:
            self._train[n] = subset_train(self.prep.train, n, seed=0)
        return self._train[n]

    def train_eval(self, n: int) -> np.ndarray:
        if n not in self._eval:
            self._eval[n] = subset_train(self.train(n), self.exp.train_eval_subset, seed=1)
        return self._eval[n]

    def init(self, n: int, k: int, seed: int) -> np.ndarray:
        key = (n, k, seed)
        if key not in self._init:
            self._init[key] = random_row_init(self.train(n), k, seed)[0]
        return self._init[key]

    def cid(self, n: int, k: int, spec_key: tuple, budget: float, seed: int) -> str:
        return make_cell_id(self.ds.name, self.ds.normalize, n, k, spec_key, budget, seed)


def _baseline_metrics(ctx: _Ctx, n: int, k: int, seed: int, key: tuple) -> tuple[float | None, float | None]:
    row = ctx.store.read_row(ctx.cid(n, k, key, ctx.exp.baseline_iterations, seed))
    if row is None or row.get("status") != "ok":
        return None, None
    return row.get("test_mse"), row.get("train_mse")


def _evaluate(ctx: _Ctx, n: int, k: int, C: np.ndarray) -> dict[str, Any]:
    tr = ctx.train_eval(n)
    labels, _ = assign(tr, C)
    stats = cluster_stats(labels, k)
    return {
        "train_mse": mse(tr, C, labels), "val_mse": mse(ctx.prep.val, C), "test_mse": mse(ctx.prep.test, C),
        "train_eval_n": int(tr.shape[0]), **stats,
    }


def _build_row(ctx: _Ctx, unit: Unit, budget: float, algo: Any, seed: int, fit: dict[str, Any] | None,
               error: str | None) -> dict[str, Any]:
    spec, n, k = unit.spec, unit.n, unit.k
    row = new_row()
    row.update(
        experiment=ctx.exp.name, cell_id=ctx.cid(n, k, spec.key, budget, seed), config_hash=ctx.cfg_hash,
        dataset=ctx.ds.name, normalize=bool(ctx.ds.normalize), n=int(n), k=int(k), d=int(ctx.ds.d),
        algo=spec.algo, backend=spec.backend, adapter=spec.adapter,
        variant=getattr(algo, "variant", "reference" if spec.backend == "reference" else None),
        params_json=canonical_json(spec.params), budget_kind=spec.budget_kind, budget=float(budget), seed=int(seed),
        role="baseline" if (spec.key == BASELINE_REF_KEY and budget == ctx.exp.baseline_iterations) else "run",
        status="ok" if error is None else "failed", error=error, keo_version=__version__, git_sha=ctx.git,
        torch_threads=ctx.torch_threads, device="cpu", peak_rss_mb=peak_rss_mb(), hardware_json=ctx.hw_json,
        timestamp_utc=utc_now(),
    )
    if fit is None:
        return row
    ev, diag, ops, tim = fit["eval"], fit["diag"], fit["ops"], fit["timings"]
    for key in ("train_eval_n", "train_mse", "val_mse", "test_mse", "dead_clusters", "empty_clusters",
                "imbalance_factor", "max_over_mean"):
        row[key] = _py(ev[key])
    ref_te, ref_tr = _baseline_metrics(ctx, n, k, seed, BASELINE_REF_KEY)
    off_te, off_tr = _baseline_metrics(ctx, n, k, seed, BASELINE_OFF_KEY)
    is_L25 = budget == ctx.exp.baseline_iterations
    if is_L25 and spec.key == BASELINE_REF_KEY:  # the baseline cell is its own reference (its row is not stored yet)
        ref_te, ref_tr = row["test_mse"], row["train_mse"]
    if is_L25 and spec.key == BASELINE_OFF_KEY:
        off_te, off_tr = row["test_mse"], row["train_mse"]
    row["rel_test_vs_lloyd25_ref"], row["rel_train_vs_lloyd25_ref"] = _rel(row["test_mse"], ref_te), _rel(row["train_mse"], ref_tr)
    row["rel_test_vs_lloyd25_official"] = _rel(row["test_mse"], off_te)
    row["rel_train_vs_lloyd25_official"] = _rel(row["train_mse"], off_tr)
    row.update(t_init=_py(tim.get("init")), t_setup=_py(tim.get("setup")), t_fit=_py(tim.get("fit")),
               t_finalize=_py(tim.get("finalize")))
    row.update(
        distance_computations=_py(ops.get("distance_computations")), assign_passes=_py(ops.get("assign_passes")),
        assign_distance_computations=_py(ops.get("assign_distance_computations")),
        maintenance_distance_computations=_py(ops.get("maintenance_distance_computations")),
    )
    row.update(
        recipe=diag.get("recipe"), batch_size=_py(diag.get("batch_size")), alpha0=_py(diag.get("alpha0")),
        n_updates=_py(diag.get("n_updates")), n_revived_total=_py(diag.get("n_revived_total")),
        nlist=_py(diag.get("nlist")), nprobe=_py(diag.get("nprobe")),
        missed_nearest_mean=_py(diag.get("missed_nearest_mean")),
        assign_fraction_actual=_py(diag.get("assign_fraction_of_exact_actual")),
        assign_fraction_nominal=_py(diag.get("assign_fraction_of_exact_nominal")),
        finish=diag.get("finish"),
    )
    return row


def _short_err(e: BaseException) -> str:
    tb = traceback.format_exception_only(type(e), e)[-1].strip()
    return tb[:500]


def _execute_unit(ctx: _Ctx, unit: Unit, pending: list[float], log: Callable[[str], None]) -> tuple[int, int]:
    spec, n, k, seed = unit.spec, unit.n, unit.k, unit.seed
    X, C0 = ctx.train(n), ctx.init(n, k, seed)
    ok = failed = 0
    fits: dict[float, dict[str, Any]] = {}
    err_all: str | None = None
    algo = None
    try:
        algo = make_algo(spec)
        group = (n, k, spec.key)
        if ctx.exp.warmup and spec.backend == "reference" and group not in ctx.warmed:
            ctx.warmed.add(group)
            try:  # discarded warm-up fit of the smallest budget (A29)
                wb = min(spec.budgets)
                algo.fit(X, k, Budget(iterations=int(wb)) if spec.algo == "lloyd" else Budget(epochs=float(wb)),
                         seed=0, init_centroids=C0)
            except Exception:
                pass
        if spec.algo == "lloyd" and spec.backend == "reference":
            top = int(max(pending))
            res = algo.fit(X, k, Budget(iterations=top), seed=seed, init_centroids=C0, checkpoints=[int(b) for b in pending])
            for b in pending:
                C, t_fit = res.diagnostics["snapshots"][int(b)]
                fits[b] = {"C": C, "timings": {"init": res.timings["init"], "fit": t_fit, "finalize": 0.0},
                           "ops": {"distance_computations": float(n) * k * b, "assign_passes": float(b)}, "diag": {}}
        else:
            for b in pending:
                try:
                    bud = Budget(iterations=int(b)) if spec.algo == "lloyd" else Budget(epochs=float(b))
                    res = algo.fit(X, k, bud, seed=seed, init_centroids=C0)
                    fits[b] = {"C": res.centroids, "timings": res.timings, "ops": res.op_counts, "diag": res.diagnostics}
                except Exception as e:
                    ctx.store.write(_build_row(ctx, unit, b, algo, seed, None, _short_err(e)))
                    failed += 1
                    log(f"    FAILED {spec.algo}/{spec.backend} budget={b}: {_short_err(e)}")
    except (AdapterUnavailable, ConfigError, TypeError, ValueError) as e:
        err_all = _short_err(e)
    except Exception as e:  # unexpected: still recorded, never silent
        err_all = _short_err(e)

    order = sorted(pending, key=lambda b: (b != ctx.exp.baseline_iterations, b)) if spec.algo == "lloyd" else list(pending)
    for b in order:
        if b not in fits:
            if err_all is not None:
                ctx.store.write(_build_row(ctx, unit, b, algo, seed, None, err_all))
                failed += 1
            continue
        try:
            fits[b]["eval"] = _evaluate(ctx, n, k, fits[b]["C"])
            ctx.store.write(_build_row(ctx, unit, b, algo, seed, fits[b], None))
            ok += 1
        except Exception as e:
            ctx.store.write(_build_row(ctx, unit, b, algo, seed, None, _short_err(e)))
            failed += 1
    return ok, failed


def run_experiment(
    exp: ExperimentConfig,
    ds_cfg: DatasetConfig,
    out_dir: str | Path,
    prepared: Prepared | None = None,
    retry_failed: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
    availability: dict[str, tuple[bool, str]] | None = None,
) -> RunSummary:
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    ds = dataclasses.replace(ds_cfg, normalize=exp.normalize) if exp.normalize is not None else ds_cfg
    if availability is None:
        availability = {s.key: (s.available, s.reason) for s in adapter_status()}
    plan = expand(exp, ds, availability)

    # fail fast on invalid reference params instead of producing a grid of failed rows
    for spec in {u.spec.key: u.spec for u in plan.units if u.spec.backend == "reference"}.values():
        try:
            make_algo(spec)
        except Exception as e:
            raise ConfigError(f"invalid params for {spec.algo}/{spec.backend}: {spec.params} -> {_short_err(e)}") from e

    store = ResultStore(out_dir)
    summary = RunSummary(out_dir=out_dir, n_cells_planned=plan.n_cells(), n_units=len(plan.units),
                         skipped=plan.skipped, dry_run=dry_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(exp.as_dict(), indent=2, default=str))
    (out_dir / "skipped.json").write_text(json.dumps(plan.skipped, indent=2, default=str))

    todo: list[tuple[Unit, list[float]]] = []
    for u in plan.units:
        pend = [b for b in u.budgets if not store.done(make_cell_id(ds.name, ds.normalize, u.n, u.k, u.spec.key, b, u.seed),
                                                         retry_failed)]
        summary.n_cells_resumed += len(u.budgets) - len(pend)
        if pend:
            todo.append((u, pend))
    if dry_run:
        summary.seconds = time.perf_counter() - t_start
        return summary

    configure_threads(exp.threads)
    hw = hardware_descriptor()
    prep = prepared if prepared is not None else prepare(ds)
    ctx = _Ctx(exp, ds, prep, store, hw)
    for s in plan.skipped:
        log(f"  skipped: {s}")
    done_units = 0
    for i, (u, pend) in enumerate(todo, 1):
        if limit is not None and done_units >= limit:
            break
        t0 = time.perf_counter()
        ok, failed = _execute_unit(ctx, u, pend, log)
        summary.n_ok += ok
        summary.n_failed += failed
        done_units += 1
        log(f"  [{i}/{len(todo)}] n={u.n} k={u.k} seed={u.seed} {u.spec.algo}/{u.spec.backend} budgets={pend}: "
            f"ok={ok} failed={failed} ({time.perf_counter() - t0:.1f}s)")
        gc.collect()
    summary.results_path = store.consolidate()
    summary.seconds = time.perf_counter() - t_start
    return summary


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Median-over-seeds table for printing: one line per (algo, backend, budget)."""
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return ok
    g = ok.groupby(["n", "k", "algo", "backend", "budget_kind", "budget"], dropna=False)
    out = g.agg(seeds=("seed", "nunique"), test_mse=("test_mse", "median"),
                rel_test_vs_L25=("rel_test_vs_lloyd25_ref", "median"), rel_train_vs_L25=("rel_train_vs_lloyd25_ref", "median"),
                passes=("assign_passes", "median"), t_fit_s=("t_fit", "median")).reset_index()
    return out
