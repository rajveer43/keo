"""Loading, dedup, normalisation and splitting. Pure-numpy helpers + a `prepare` driver.

Protocol follows e-means §4.1 "Scoring": held-out vectors are deduplicated against the training set
to avoid contamination, then split 50/50 into validation / test (A25).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from keo.config import DatasetConfig
from keo.data.download import dataset_file


@dataclass
class Prepared:
    train: np.ndarray  # float32 [n, d]
    val: np.ndarray
    test: np.ndarray
    report: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- hashing / dedup
def row_hashes(X: np.ndarray) -> list[bytes]:
    """128-bit blake2b digest of each row's raw bytes (exact-match hashing)."""
    X = np.ascontiguousarray(X)
    return [hashlib.blake2b(X[i].tobytes(), digest_size=16).digest() for i in range(X.shape[0])]


def dedup_eval_against_train(train: np.ndarray, eval_: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Return indices of eval rows to keep (not present in train, first occurrence only)."""
    train_set = set(row_hashes(train))
    keep: list[int] = []
    seen: set[bytes] = set()
    removed_in_train = removed_internal = 0
    for i, h in enumerate(row_hashes(eval_)):
        if h in train_set:
            removed_in_train += 1
        elif h in seen:
            removed_internal += 1
        else:
            seen.add(h)
            keep.append(i)
    report = {
        "eval_in": int(eval_.shape[0]),
        "eval_removed_present_in_train": removed_in_train,
        "eval_removed_internal_duplicates": removed_internal,
        "eval_kept": len(keep),
    }
    return np.asarray(keep, dtype=np.int64), report


def count_internal_duplicates(X: np.ndarray) -> int:
    hs = row_hashes(X)
    return len(hs) - len(set(hs))


# ---------------------------------------------------------------- normalisation
def norm_stats(X: np.ndarray) -> dict[str, float]:
    n = np.linalg.norm(X.astype(np.float64), axis=1)
    return {
        "min": float(n.min()),
        "p01": float(np.percentile(n, 1)),
        "median": float(np.median(n)),
        "mean": float(n.mean()),
        "p99": float(np.percentile(n, 99)),
        "max": float(n.max()),
        "frac_within_1e-3_of_1": float(np.mean(np.abs(n - 1.0) < 1e-3)),
        "n_zero": int(np.sum(n == 0.0)),
    }


def l2_normalize(X: np.ndarray, eps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalise to unit L2 norm in float64 then cast to float32.

    Returns (Xn, nonzero_mask). Rows with norm <= eps are NOT normalised and are flagged False.
    """
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    mask = norms > eps
    out = np.zeros_like(X, dtype=np.float32)
    out[mask] = (X[mask].astype(np.float64) / norms[mask, None]).astype(np.float32)
    return out, mask


# ---------------------------------------------------------------- splitting
def split_val_test(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return np.sort(perm[:n_val]), np.sort(perm[n_val:])


def subset_train(X: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Fixed-seed random subset of train rows (n=None -> all rows). Order is preserved (sorted indices)."""
    if n is None or n >= X.shape[0]:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=n, replace=False))
    return X[idx]


# ---------------------------------------------------------------- driver
def load_raw(cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    path = dataset_file(cfg)
    with h5py.File(path, "r") as f:
        train = np.asarray(f[cfg.source.train_key], dtype=np.float32)
        eval_ = np.asarray(f[cfg.source.eval_key], dtype=np.float32)
    return train, eval_


def prepare(cfg: DatasetConfig) -> Prepared:
    train_raw, eval_raw = load_raw(cfg)
    rep: dict[str, Any] = {"dataset": cfg.name}

    if train_raw.shape != (cfg.n_train, cfg.d):
        raise ValueError(f"train shape {train_raw.shape} != config ({cfg.n_train}, {cfg.d})")
    if eval_raw.shape != (cfg.n_eval, cfg.d):
        raise ValueError(f"eval shape {eval_raw.shape} != config ({cfg.n_eval}, {cfg.d})")
    if not (np.isfinite(train_raw).all() and np.isfinite(eval_raw).all()):
        raise ValueError("non-finite values in dataset")

    rep["train_shape"] = tuple(train_raw.shape)
    rep["eval_shape"] = tuple(eval_raw.shape)
    rep["raw_norm_stats_train"] = norm_stats(train_raw)
    rep["raw_norm_stats_eval"] = norm_stats(eval_raw)
    rep["train_internal_duplicates"] = count_internal_duplicates(train_raw)

    # Dedup on the RAW bytes (before any preprocessing), as exact-match hashing.
    keep, dd = dedup_eval_against_train(train_raw, eval_raw)
    rep["dedup"] = dd
    eval_raw = eval_raw[keep]

    train, eval_ = train_raw, eval_raw
    rep["normalize"] = cfg.normalize
    if cfg.normalize:
        train, m_tr = l2_normalize(train_raw)
        eval_, m_ev = l2_normalize(eval_raw)
        rep["zero_norm_dropped"] = {"train": int((~m_tr).sum()), "eval": int((~m_ev).sum())}
        train, eval_ = train[m_tr], eval_[m_ev]

    val_idx, test_idx = split_val_test(eval_.shape[0], cfg.split.val_fraction, cfg.split.seed)
    val, test = eval_[val_idx], eval_[test_idx]
    rep["sizes"] = {"train": int(train.shape[0]), "val": int(val.shape[0]), "test": int(test.shape[0])}
    return Prepared(train=train, val=val, test=test, report=rep)
