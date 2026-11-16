"""[O] o-means, reference backend (PyTorch). Spec: docs/SPEC.md §5; source: o-means project page [O §03-04].

o-means = e-means (same objective, random-row init, EMA update, both laws, cosine schedule, revival: this class
literally subclasses `EMeans` and reuses its training loop) with three changes:
  1. assignment: two-level IVF over the centroids (`ivf_cells.CellIndex`), kept current by
       - refresh every 5 batches during the first epoch (after the revival check),
       - rebuild every half epoch after the first epoch,
       - split pointers donor -> child (+ grandchild check);
  2. finishing: for T < 2 a recency-weighted closing mean; for T >= 2 e-means' plain last-epoch mean;
  3. kernels (NOT reproduced here: this backend is for correctness and operation counts; wall-clock claims about the
     GPU layout need the authors' CUDA implementation).

The paper is unreleased; everything the project page leaves open is recorded in docs/ASSUMPTIONS.md (A14-A22).
`OMeans.assign_exact` is the exact search the page calls `assign(x)`; all reported MSE uses it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from keo.algos.emeans import EMeans
from keo.algos.ivf_cells import DEFAULT_CELL_SIZE, CellIndex
from keo.metrics.core import assign


class OMeansAssigner:
    def __init__(
        self,
        k: int,
        d: int,
        nb: int,
        seed_seq: np.random.SeedSequence,
        nlist: int | None,
        nprobe: int | None,
        cell_size: int,
        kmeans_iters: int,
        refresh_every: int,
        probe_frac: float,
        maintain: bool = True,
        pointers: bool = True,
    ) -> None:
        build_ss, probe_ss = seed_seq.spawn(2)
        self.index = CellIndex(
            k, d, nlist=nlist, nprobe=nprobe, cell_size=cell_size, kmeans_iters=kmeans_iters,
            seed=int(build_ss.generate_state(1)[0] % (2**31 - 1)),
        )
        self.k, self.nb = k, nb
        self.refresh_every, self.probe_frac = refresh_every, probe_frac
        self.maintain, self.pointers = maintain, pointers
        self.rng_probe = np.random.default_rng(probe_ss)
        self.assign_ops = 0.0  # level-1 + level-2 + pointer checks (actual list sizes)
        self.nominal_ops = 0.0  # B*(nlist + nprobe*cell_size)
        self.exact_ops = 0.0  # B*k, what brute force would have cost
        self.level_ops = {"cells": 0.0, "centroids": 0.0, "pointer": 0.0}
        self.pairs_visited = 0.0
        self.missed: list[float] = []
        self.missed_excess: list[float] = []
        self.refresh_batches: list[int] = []
        self.rebuild_batches: list[int] = []

    @property
    def maintenance_ops(self) -> float:
        return self.index.maintenance_ops

    def start(self, C: torch.Tensor) -> None:
        self.index.build(C)

    def assign(self, xb: torch.Tensor, C: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels, d2, ops = self.index.search(xb, C)
        B = xb.shape[0]
        self.assign_ops += ops["cells"] + ops["centroids"] + ops["pointer"]
        self.nominal_ops += ops["nominal"]
        self.exact_ops += float(B) * self.k
        for key in self.level_ops:
            self.level_ops[key] += ops[key]
        self.pairs_visited += float(B) * min(self.index.nprobe, int((self.index.cell_sizes() > 0).sum()))
        self._probe(xb, C, labels, d2)
        return labels, d2

    def _probe(self, xb: torch.Tensor, C: torch.Tensor, labels: torch.Tensor, d2: torch.Tensor) -> None:
        """Missed-nearest rate on a sample of the batch (cost excluded from op counts; A22)."""
        B = xb.shape[0]
        m = min(B, max(256, int(self.probe_frac * B)))
        s = torch.from_numpy(self.rng_probe.choice(B, size=m, replace=False))
        lab_e, d2_e = assign(xb[s], C)
        missed = (lab_e != labels[s]) & (d2[s] > d2_e + 1e-5 * (1.0 + d2_e))
        self.missed.append(float(missed.double().mean()))
        self.missed_excess.append(float(((d2[s] - d2_e) / (d2_e + 1e-12)).clamp_min(0).double().mean()))

    def on_revival(self, pairs: list[tuple[int, int]]) -> None:
        if not self.maintain:  # static-index ablation: the cells are built once and never touched again
            return
        for dead, donor in pairs:
            self.index.note_split(dead, donor, pointer=self.pointers)

    def after_batch(self, t: int, nb: int, get_C: Callable[[], torch.Tensor]) -> None:
        if not self.maintain:
            return
        u = t + 1  # updates completed
        if u <= nb:  # first epoch: refresh on the revival cadence (A16)
            do = (u % self.refresh_every == 0) if nb >= self.refresh_every else (u == nb)
            if do:
                self.index.refresh(get_C())
                self.refresh_batches.append(u)
        else:  # afterwards: rebuild every half epoch
            h = max(1, nb // 2)
            if (u - nb) % h == 0:
                self.index.rebuild(get_C())
                self.rebuild_batches.append(u)

    def diagnostics(self) -> dict[str, Any]:
        sizes = self.index.cell_sizes().double()
        mean_actual_cs = self.level_ops["centroids"] / self.pairs_visited if self.pairs_visited else float("nan")
        return {
            "nlist": self.index.nlist,
            "nprobe": self.index.nprobe,
            "cell_size_nominal": self.index.cell_size,
            "cell_size_actual_visited_mean": mean_actual_cs,
            "cell_size_final": {
                "min": float(sizes.min()), "mean": float(sizes.mean()), "max": float(sizes.max()),
                "empty_cells": int((sizes == 0).sum()),
            },
            "n_refresh": self.index.n_refresh,
            "n_rebuild": self.index.n_rebuild,
            "refresh_batches": self.refresh_batches,
            "rebuild_batches": self.rebuild_batches,
            "n_split_pointers": int((self.index.child >= 0).sum()),
            "missed_nearest_per_batch": self.missed,
            "missed_nearest_mean": float(np.mean(self.missed)) if self.missed else float("nan"),
            "missed_nearest_excess_dist_mean": float(np.mean(self.missed_excess)) if self.missed_excess else float("nan"),
            "assign_fraction_of_exact_actual": self.assign_ops / self.exact_ops if self.exact_ops else float("nan"),
            "assign_fraction_of_exact_nominal": self.nominal_ops / self.exact_ops if self.exact_ops else float("nan"),
            "assign_ops_by_level": dict(self.level_ops),
        }


class OMeans(EMeans):
    name = "omeans"
    backend = "reference"

    def __init__(
        self,
        nlist: int | None = None,
        nprobe: int | None = None,
        cell_size: int = DEFAULT_CELL_SIZE,
        cell_kmeans_iters: int = 10,
        refresh_every: int = 5,
        probe_frac: float = 0.01,
        recency_below_epochs: float = 2.0,
        recency_tau_epochs: float = 1.0,
        maintain: bool = True,  # False: static index (build once; no refresh/rebuild/split filing) [ablation, O §04.2]
        pointers: bool = True,  # False: keep refresh/rebuild but drop donor->child pointers   [ablation]
        **emeans_kwargs: Any,
    ) -> None:
        super().__init__(**emeans_kwargs)
        self.maintain, self.pointers = maintain, pointers
        self.nlist, self.nprobe, self.cell_size = nlist, nprobe, cell_size
        self.cell_kmeans_iters, self.refresh_every, self.probe_frac = cell_kmeans_iters, refresh_every, probe_frac
        self.recency_below_epochs, self.recency_tau_epochs = recency_below_epochs, recency_tau_epochs

    def _new_assigner(self, k: int, d: int, nb: int, seed_seq: np.random.SeedSequence) -> OMeansAssigner:  # type: ignore[override]
        return OMeansAssigner(
            k, d, nb, seed_seq, self.nlist, self.nprobe, self.cell_size,
            self.cell_kmeans_iters, self.refresh_every, self.probe_frac, self.maintain, self.pointers,
        )

    def _use_recency(self, T: float) -> bool:
        return self.finalize == "last_epoch_mean" and T < self.recency_below_epochs  # A19

    def _recency_tau(self) -> float:
        return self.recency_tau_epochs

    @staticmethod
    def assign_exact(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact nearest-centroid search (the page's `assign(x)`)."""
        lab, d2 = assign(X, centroids)
        return lab.numpy(), d2.numpy()
