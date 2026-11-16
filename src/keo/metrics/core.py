"""Exact assignment and clustering metrics.

All reported MSE values use EXACT nearest-centroid assignment [E §4.1 "Scoring"] and are recomputed
in float64 for the assigned pair (A5). MSE is the mean over points of the squared L2 distance summed
over coordinates [E Eq.1] (A23).
"""

from __future__ import annotations

import numpy as np
import torch

DEAD_THRESHOLD = 10  # a cluster is "dead" if < 10 database vectors are assigned [E App. H.2]
DEFAULT_CHUNK_ELEMS = 1 << 25  # rows*k elements of the score matrix per chunk (~128 MB fp32)


def as_tensor(X: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return X
    return torch.from_numpy(np.ascontiguousarray(X))


def assign(
    X: np.ndarray | torch.Tensor,
    C: np.ndarray | torch.Tensor,
    chunk_elems: int = DEFAULT_CHUNK_ELEMS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact nearest centroid. Returns (labels int64 [n], min squared distance float32 [n]).

    Uses ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2 in fp32 with a chunked score matrix (never n x k).
    """
    X, C = as_tensor(X), as_tensor(C)
    n, k = X.shape[0], C.shape[0]
    cnorm = (C * C).sum(1)
    labels = torch.empty(n, dtype=torch.int64)
    d2 = torch.empty(n, dtype=torch.float32)
    rows = max(1, chunk_elems // max(k, 1))
    Ct = C.T
    for s in range(0, n, rows):
        xb = X[s : s + rows]
        scores = torch.addmm(cnorm[None, :], xb, Ct, alpha=-2.0)  # ||c||^2 - 2 x.c
        v, idx = scores.min(dim=1)
        labels[s : s + rows] = idx
        d2[s : s + rows] = (v + (xb * xb).sum(1)).clamp_min_(0.0)
    return labels, d2


def mse(
    X: np.ndarray | torch.Tensor,
    C: np.ndarray | torch.Tensor,
    labels: torch.Tensor | None = None,
    chunk_rows: int = 65536,
) -> float:
    """Exact-assignment MSE, final distance recomputed in float64."""
    X, C = as_tensor(X), as_tensor(C)
    if labels is None:
        labels, _ = assign(X, C)
    total = 0.0
    for s in range(0, X.shape[0], chunk_rows):
        diff = X[s : s + chunk_rows].double() - C[labels[s : s + chunk_rows]].double()
        total += float((diff * diff).sum())
    return total / X.shape[0]


def cluster_stats(labels: torch.Tensor | np.ndarray, k: int) -> dict[str, float]:
    """Cluster-size diagnostics.

    - dead: clusters with < 10 assigned points [E App. H.2]
    - empty: clusters with 0 points
    - imbalance_factor: k * sum(size^2) / n^2 (Faiss definition; 1.0 = perfectly balanced). This is our
      reading of "imbalance" in [E Table 14] (values 1.16-1.65); the paper does not define it (A35).
    - max_over_mean: largest cluster / (n/k)
    """
    lab = torch.as_tensor(labels, dtype=torch.int64)
    sizes = torch.bincount(lab, minlength=k).double()
    n = float(sizes.sum())
    return {
        "dead_clusters": int((sizes < DEAD_THRESHOLD).sum()),
        "empty_clusters": int((sizes == 0).sum()),
        "imbalance_factor": float(k * (sizes * sizes).sum() / (n * n)),
        "max_over_mean": float(sizes.max() / (n / k)),
    }
