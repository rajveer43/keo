"""Common interface for K / E / O clusterers (reference and official backends)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Budget:
    """Training budget. Lloyd uses `iterations`; E/O use `epochs` (may be fractional)."""

    epochs: float | None = None
    iterations: int | None = None

    def __post_init__(self) -> None:
        if (self.epochs is None) == (self.iterations is None):
            raise ValueError("Budget needs exactly one of epochs / iterations")
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.iterations is not None and self.iterations < 1:
            raise ValueError("iterations must be >= 1")


@dataclass
class FitResult:
    centroids: np.ndarray  # float32 [k, d]
    train_labels: np.ndarray | None  # int64 [n]; may be approximate/stale (documented per algorithm)
    timings: dict[str, float] = field(default_factory=dict)  # seconds: init, fit, finalize
    op_counts: dict[str, float] = field(default_factory=dict)  # distance_computations, assign_passes
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Clusterer(Protocol):
    name: str  # "lloyd" | "emeans" | "omeans"
    backend: str  # "reference" | "official"

    def fit(self, X: np.ndarray, k: int, budget: Budget, seed: int) -> FitResult: ...
