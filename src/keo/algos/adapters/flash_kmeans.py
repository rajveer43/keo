"""Official backend: Flash-KMeans (Yang et al., arXiv:2603.09229), the GPU Lloyd reference of both papers.

UNVERIFIED adapter: the package (PyPI `flash-kmeans`, 0.3.1 in both papers) is not installed here and needs CUDA/Triton.
Written from its public README (`batch_kmeans_Euclid(x[B,N,D], n_clusters, tol, verbose, ...)` -> (ids, centers, _)) and from
[E App. C] ("one public-API call, forced to the full count: max_iters=T, tol=-1"). Because parameter names are not
confirmed, the call passes only keyword arguments that appear in the function's signature, and records which were
honoured. `flash-kmeans-mlx` (an independent Apple-Silicon port by someone else) is NOT this implementation and is
deliberately not used.
"""

from __future__ import annotations

import importlib.util
import inspect
import time

import numpy as np

from keo.algos.adapters._worker_client import AdapterUnavailable
from keo.algos.base import Budget, FitResult
from keo.algos.init import random_row_init


def is_available() -> tuple[bool, str]:
    if importlib.util.find_spec("flash_kmeans") is None:
        return False, "flash_kmeans is not installed (pip install flash-kmeans; needs a CUDA GPU with Triton)"
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, "flash_kmeans needs a CUDA GPU; none is visible"
    return True, "flash_kmeans + CUDA (UNVERIFIED adapter)"


class FlashKMeans:
    name = "lloyd"
    backend = "official"
    variant = "flash-kmeans (UNVERIFIED)"

    def __init__(self, dtype: str = "float32") -> None:
        self.dtype = dtype

    def fit(self, X: np.ndarray, k: int, budget: Budget, seed: int,
            init_centroids: np.ndarray | None = None) -> FitResult:
        ok, why = is_available()
        if not ok:
            raise AdapterUnavailable(why)
        import flash_kmeans
        import torch

        iters = budget.iterations if budget.iterations is not None else int(budget.epochs)  # type: ignore[arg-type]
        n, d = X.shape
        fn = flash_kmeans.batch_kmeans_Euclid
        sig = inspect.signature(fn).parameters
        wanted = {"n_clusters": k, "max_iters": iters, "tol": -1.0, "verbose": False}
        if init_centroids is not None:
            wanted["init_centroids"] = torch.from_numpy(np.ascontiguousarray(init_centroids))[None].cuda()
        kwargs = {kk: v for kk, v in wanted.items() if kk in sig}
        honoured, ignored = sorted(kwargs), sorted(set(wanted) - set(kwargs))
        torch.manual_seed(seed)
        x = torch.from_numpy(np.ascontiguousarray(X)).to(getattr(torch, self.dtype))[None].cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, centers, *_ = fn(x, **kwargs)
        torch.cuda.synchronize()
        fit_s = time.perf_counter() - t0
        return FitResult(
            centroids=centers[0].float().cpu().numpy(),
            train_labels=None,
            timings={"init": 0.0, "fit": fit_s, "finalize": 0.0},
            op_counts={"distance_computations": float(n) * k * iters, "assign_passes": float(iters)},
            diagnostics={"init": "given" if "init_centroids" in kwargs else "own", "iterations": iters,
                         "kwargs_honoured": honoured, "kwargs_ignored_not_in_signature": ignored,
                         "shared_init": "init_centroids" in kwargs, "variant": self.variant},
        )
