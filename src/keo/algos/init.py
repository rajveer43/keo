"""Centroid initialisation shared by K, E and O so that one seed means one identical start.

Random-row init: k distinct rows of the training set, drawn without replacement [E §4.1 "Initialization"].
k-means++: scikit-learn's greedy variant, the strong one per [E App. G.2] (Faiss's plain k-means++ gave
"null to negative" gain). Its cost is reported separately by callers.
"""

from __future__ import annotations

import numpy as np


def random_row_init(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return np.ascontiguousarray(X[idx], dtype=np.float32), idx


def kmeanspp_init(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import kmeans_plusplus

    centers, idx = kmeans_plusplus(np.asarray(X), n_clusters=k, random_state=seed)
    return np.ascontiguousarray(centers, dtype=np.float32), np.asarray(idx)
