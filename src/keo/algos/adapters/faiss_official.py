"""Official backends shipped by the authors.

* `OfficialEmeans`  -- `faiss.contrib.e_means.NumpyEmeans` (in faiss-cpu >= 1.16.0). VERIFIED on this machine: the API was
  read from the installed package (constructor, `train(x, init_centroids=...)`, `recipe`), and it runs on CPU.
* `OfficialOmeans`  -- `faiss.contrib.o_means.Omeans` from the o-means authors' faiss fork. NOT verifiable here: no public
  URL for the fork could be found, it is not in the released wheel, and it needs CUDA. Written from the project page
  only (`Omeans(d, k, nepochs)`, `.train(x)`), so it is flagged UNVERIFIED and reports "unavailable" cleanly.

Both run in the isolated worker process (A36). Neither is allowed to be mixed with `reference` rows (backend column).
"""

from __future__ import annotations

import time

import numpy as np

from keo.algos.adapters._worker_client import AdapterUnavailable, faiss_installed, probe, run_worker
from keo.algos.base import Budget, FitResult
from keo.algos.init import random_row_init


def emeans_available() -> tuple[bool, str]:
    if not faiss_installed():
        return False, "faiss is not installed (pip install faiss-cpu)"
    p = probe()
    if not p.get("has_e_means"):
        return False, f"faiss {p.get('faiss_version')} has no contrib.e_means (needs >= 1.16.0)"
    return True, f"faiss {p['faiss_version']} contrib.e_means.NumpyEmeans (CPU, isolated worker)"


def omeans_available() -> tuple[bool, str]:
    if not faiss_installed():
        return False, "faiss is not installed"
    p = probe()
    if not p.get("has_o_means"):
        return False, ("faiss.contrib.o_means is not in the installed faiss "
                       f"({p.get('faiss_version')}): it lives in the o-means authors' fork (no public URL found); needs CUDA")
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "o-means is a GPU algorithm and no CUDA device is visible"
    except ImportError:
        return False, "torch is not installed"
    return True, "faiss fork with contrib.o_means + CUDA (UNVERIFIED adapter)"


class OfficialEmeans:
    name = "emeans"
    backend = "official"
    variant = "faiss.contrib.e_means.NumpyEmeans"

    def __init__(self, threads: int | None = None, batch_law: str = "auto", unused_threshold: float | None = None,
                 batch_size: int | None = None, alpha0: float | None = None) -> None:
        self.threads, self.batch_law, self.unused_threshold = threads, batch_law, unused_threshold
        self.batch_size, self.alpha0 = batch_size, alpha0

    def fit(self, X: np.ndarray, k: int, budget: Budget, seed: int,
            init_centroids: np.ndarray | None = None) -> FitResult:
        ok, why = emeans_available()
        if not ok:
            raise AdapterUnavailable(why)
        if budget.epochs is None:
            raise ValueError("OfficialEmeans needs an epochs budget")
        n, d = X.shape
        t0 = time.perf_counter()
        if init_centroids is None:
            init_centroids, _ = random_row_init(X, k, seed)
        t_init = time.perf_counter() - t0

        res, centroids, labels = run_worker(
            {"algo": "emeans", "k": k, "epochs": float(budget.epochs), "seed": seed, "threads": self.threads,
             "batch_law": self.batch_law, "unused_threshold": self.unused_threshold,
             "batch_size": self.batch_size, "alpha0": self.alpha0},
            X=X, init=init_centroids, want_labels=True,
        )
        assert centroids is not None
        rec = res["recipe"]
        rows = float(sum(rec.get("epoch_rows", [])))
        dist = rows * k
        return FitResult(
            centroids=centroids,
            train_labels=labels,
            timings={"init": t_init, "fit": float(res["fit_seconds"]), "finalize": 0.0},
            op_counts={"distance_computations": dist, "assign_passes": dist / (float(n) * k)},
            diagnostics={
                "init": "given", "epochs": float(budget.epochs), "n": n, "k": k,
                "batch_size": rec.get("batch_size"), "alpha0": rec.get("alpha0"), "batch_law": rec.get("batch_law"),
                "steps_per_pass": rec.get("steps_per_pass"), "n_updates": rec.get("total_steps"),
                "n_revived_total": res["n_splits"], "recipe_official": rec,
                "faiss_version": res["faiss_version"], "variant": self.variant,
            },
        )


class OfficialOmeans:
    """UNVERIFIED adapter (see module docstring). Cannot share the reference init: the page documents no init argument."""

    name = "omeans"
    backend = "official"
    variant = "faiss-fork contrib.o_means.Omeans (UNVERIFIED)"

    def fit(self, X: np.ndarray, k: int, budget: Budget, seed: int,
            init_centroids: np.ndarray | None = None) -> FitResult:
        ok, why = omeans_available()
        if not ok:
            raise AdapterUnavailable(why)
        if budget.epochs is None:
            raise ValueError("OfficialOmeans needs an epochs budget")
        n, d = X.shape
        res, centroids, _ = run_worker({"algo": "omeans", "k": k, "epochs": float(budget.epochs), "seed": seed}, X=X)
        assert centroids is not None
        return FitResult(
            centroids=centroids, train_labels=None,
            timings={"init": 0.0, "fit": float(res["fit_seconds"]), "finalize": 0.0},
            op_counts={},  # the fork does not expose operation counts
            diagnostics={"init": "own (random rows of x; not shared)", "shared_init": False, "variant": self.variant,
                         "ignored_init_centroids": init_centroids is not None},
        )
