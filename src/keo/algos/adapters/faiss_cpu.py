"""Official backend: Faiss Lloyd k-means (the e-means paper's Lloyd reference, [E App. C]); CPU, or GPU if available.

Runs in a SEPARATE PROCESS (see _faiss_worker.py): this parent process must never import faiss, because the
torch and faiss wheels ship conflicting libomp copies on macOS (A36). Reported fit time is measured inside the
worker around `Kmeans.train` only, so data serialisation / process start-up are excluded.

Faiss's automatic sub-sampling (max 256 points per centroid) is disabled, as in the paper [E §4.1 fn.1].
`niter` = number of Lloyd iterations (verified: one objective entry per iteration).
"""

from __future__ import annotations

import importlib.metadata
import time

import numpy as np

from keo.algos.adapters._worker_client import AdapterUnavailable, faiss_installed, probe, run_worker
from keo.algos.base import Budget, FitResult
from keo.algos.init import random_row_init


def is_available(gpu: bool = False) -> tuple[bool, str]:
    """Check availability WITHOUT importing faiss into this process."""
    if not faiss_installed():
        return False, "faiss is not installed (pip install faiss-cpu)"
    try:
        ver = importlib.metadata.version("faiss-cpu")
    except importlib.metadata.PackageNotFoundError:
        ver = "unknown"
    if gpu:
        n = int(probe().get("num_gpus", 0))
        return (n > 0, f"faiss-cpu {ver}; {n} GPU(s) visible to faiss" if n else
                f"faiss-cpu {ver} has no GPU support / no GPU visible (needs faiss-gpu on a CUDA machine)")
    return True, f"faiss-cpu {ver} (isolated worker process)"


class FaissLloyd:
    name = "lloyd"

    def __init__(self, threads: int | None = None, gpu: bool = False) -> None:
        self.threads, self.gpu = threads, gpu
        self.backend = "official"
        self.variant = "faiss-gpu" if gpu else "faiss-cpu"

    def fit(
        self,
        X: np.ndarray,
        k: int,
        budget: Budget,
        seed: int,
        init_centroids: np.ndarray | None = None,
    ) -> FitResult:
        ok, why = is_available(self.gpu)
        if not ok:
            raise AdapterUnavailable(why)
        iters = budget.iterations if budget.iterations is not None else int(budget.epochs)  # type: ignore[arg-type]
        n, d = X.shape

        t0 = time.perf_counter()
        if init_centroids is None:
            init_centroids, _ = random_row_init(X, k, seed)
        t_init = time.perf_counter() - t0

        res, centroids, _ = run_worker(
            {"algo": "kmeans", "k": k, "niter": iters, "seed": seed, "threads": self.threads, "gpu": self.gpu},
            X=X, init=init_centroids,
        )
        assert centroids is not None
        obj = np.asarray(res["obj"], dtype=np.float64)  # sum of squared distances at each iteration's assignment
        return FitResult(
            centroids=centroids,
            train_labels=None,
            timings={"init": t_init, "fit": float(res["fit_seconds"]), "finalize": 0.0},
            op_counts={"distance_computations": float(n) * k * iters, "assign_passes": float(iters)},
            diagnostics={
                "init": "given",
                "iterations": iters,
                "train_mse_per_iter": (obj / res["n"]).tolist(),
                "faiss_version": res["faiss_version"],
                "faiss_omp_threads": res["omp_threads"],
                "variant": self.variant,
            },
        )
