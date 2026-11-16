"""[K] Lloyd's k-means, reference backend (PyTorch, CPU/any device).

Spec: docs/SPEC.md §3. Per iteration: exact assignment of all n points, then every centroid becomes the
mean of its points [E §1]. Empty clusters follow `empty_policy`:

- "split" (default, A13): Faiss-style. Verified empirically against faiss 1.16.0 in Phase 2 (see
  docs/ASSUMPTIONS.md A13): an empty slot receives a copy of a donor centroid chosen with probability
  proportional to (count-1); the pair is perturbed symmetrically by (1 +/- 1/1024) on alternating
  coordinates and the donor's count is halved.
- "keep": the empty centroid keeps its previous position.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

from keo.algos.base import Budget, FitResult
from keo.algos.init import kmeanspp_init, random_row_init
from keo.metrics.core import DEFAULT_CHUNK_ELEMS, as_tensor, assign

FAISS_EPS = 1.0 / 1024.0
_SUM_CHUNK_ROWS = 65536


def _split_empty(
    C: torch.Tensor, counts: torch.Tensor, rng: np.random.Generator
) -> tuple[int, int]:
    """Faiss-style empty-cluster splitting, in place on C. Returns (n_split, n_skipped)."""
    h = counts.clone().double().numpy()
    empties = np.flatnonzero(h == 0)
    n_split = n_skipped = 0
    d = C.shape[1]
    even = torch.arange(d) % 2 == 0
    up = torch.where(even, 1.0 + FAISS_EPS, 1.0 - FAISS_EPS).to(C.dtype)
    down = torch.where(even, 1.0 - FAISS_EPS, 1.0 + FAISS_EPS).to(C.dtype)
    for ci in empties:
        w = np.maximum(h - 1.0, 0.0)
        tot = w.sum()
        if tot <= 0:
            n_skipped += 1
            continue
        cj = int(rng.choice(len(h), p=w / tot))
        donor = C[cj].clone()
        C[ci] = donor * up
        C[cj] = donor * down
        h[cj] /= 2.0
        h[ci] = h[cj]
        n_split += 1
    return n_split, n_skipped


class Lloyd:
    name = "lloyd"
    backend = "reference"

    def __init__(
        self,
        init: str = "random",
        empty_policy: str = "split",
        chunk_elems: int = DEFAULT_CHUNK_ELEMS,
    ) -> None:
        if init not in {"random", "kmeanspp"}:
            raise ValueError(f"init must be 'random' or 'kmeanspp', got {init!r}")
        if empty_policy not in {"split", "keep"}:
            raise ValueError(f"empty_policy must be 'split' or 'keep', got {empty_policy!r}")
        self.init = init
        self.empty_policy = empty_policy
        self.chunk_elems = chunk_elems

    def fit(
        self,
        X: np.ndarray,
        k: int,
        budget: Budget,
        seed: int,
        init_centroids: np.ndarray | None = None,
        checkpoints: Sequence[int] = (),
    ) -> FitResult:
        """Run Lloyd for `budget.iterations` iterations (an `epochs` budget is read as iterations).

        `checkpoints`: iteration numbers (<= total) at which to snapshot centroids and cumulative fit time;
        Lloyd is deterministic given its init, so a snapshot at iteration i equals an independent run with
        iterations=i. Snapshots are returned in diagnostics["snapshots"] = {i: (centroids, fit_seconds)}.
        """
        iters = budget.iterations if budget.iterations is not None else int(budget.epochs)  # type: ignore[arg-type]
        if budget.iterations is None and float(budget.epochs) != iters:  # type: ignore[arg-type]
            raise ValueError("Lloyd needs an integer number of iterations")
        n, d = X.shape
        Xt = as_tensor(X)

        t0 = time.perf_counter()
        if init_centroids is not None:
            C0 = np.ascontiguousarray(init_centroids, dtype=np.float32)
            init_name = "given"
        elif self.init == "kmeanspp":
            C0, _ = kmeanspp_init(X, k, seed)
            init_name = "kmeanspp"
        else:
            C0, _ = random_row_init(X, k, seed)
            init_name = "random"
        t_init = time.perf_counter() - t0

        C = torch.from_numpy(C0.copy())
        rng = np.random.default_rng(seed * 7919 + 123)
        ckpts = set(int(c) for c in checkpoints)
        snapshots: dict[int, tuple[np.ndarray, float]] = {}
        train_mse_per_iter: list[float] = []
        empty_per_iter: list[int] = []
        n_split_total = n_skipped_total = 0
        labels = torch.empty(0, dtype=torch.int64)
        fit_time = 0.0

        for it in range(1, iters + 1):
            ts = time.perf_counter()
            labels, d2 = assign(Xt, C, self.chunk_elems)
            train_mse_per_iter.append(float(d2.double().mean()))  # objective at assignment time (pre-update)

            sums = torch.zeros(k, d, dtype=torch.float64)
            for s in range(0, n, _SUM_CHUNK_ROWS):
                sums.index_add_(0, labels[s : s + _SUM_CHUNK_ROWS], Xt[s : s + _SUM_CHUNK_ROWS].double())
            counts = torch.bincount(labels, minlength=k)
            nonempty = counts > 0
            C[nonempty] = (sums[nonempty] / counts[nonempty, None].double()).float()
            n_empty = int((~nonempty).sum())
            empty_per_iter.append(n_empty)
            if n_empty and self.empty_policy == "split":
                ns, nk = _split_empty(C, counts, rng)
                n_split_total += ns
                n_skipped_total += nk
            fit_time += time.perf_counter() - ts

            if it in ckpts:
                snapshots[it] = (C.numpy().copy(), fit_time)

        return FitResult(
            centroids=C.numpy().copy(),
            train_labels=labels.numpy().copy(),  # labels of the LAST assignment (one update stale)
            timings={"init": t_init, "fit": fit_time, "finalize": 0.0},
            op_counts={"distance_computations": float(n) * k * iters, "assign_passes": float(iters)},
            diagnostics={
                "init": init_name,
                "empty_policy": self.empty_policy,
                "iterations": iters,
                "train_mse_per_iter": train_mse_per_iter,
                "empty_clusters_per_iter": empty_per_iter,
                "n_split_total": n_split_total,
                "n_split_skipped": n_skipped_total,
                "snapshots": snapshots,
            },
        )
