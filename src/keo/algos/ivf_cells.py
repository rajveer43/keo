"""Two-level (IVF-over-centroids) nearest-centroid search for o-means. Source: [O §04.1-04.3].

The `k` centroids are grouped into `nlist` cells of about `cell_size` (128) centroids by a small k-means run over the
centroids. To assign a point: (1) distances to the `nlist` cell centres, keep the `nprobe` nearest cells;
(2) distances to the centroids inside those cells only; (3) minimum; (4) if the winner has been split, also check its
child and grandchild.

Reference-backend notes (docs/ASSUMPTIONS.md A14-A18):
  * Cell MEMBERSHIP is only updated by `refresh` / `rebuild` / `note_split`; centroid VALUES are always current
    (the sorted table is re-gathered from the live centroids each batch). That is exactly the staleness the
    maintenance schedule exists to bound.
  * Cells are unbalanced (a k-means never yields exactly 128); lists have their true variable length (A15).
  * Plain PyTorch gathers; the bucketed per-cell dense products mirror the paper's kernel structure but this is NOT
    the paper's GPU kernel, so wall-clock here says nothing about it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from keo.algos.base import Budget
from keo.algos.lloyd import Lloyd
from keo.metrics.core import assign

DEFAULT_CELL_SIZE = 128


def default_nlist(k: int, cell_size: int = DEFAULT_CELL_SIZE) -> int:
    """ceil(k / cell_size) rounded UP to a multiple of 16 [O §04 'At a glance']."""
    return int(math.ceil(math.ceil(k / cell_size) / 16) * 16)


def default_nprobe(k: int, d: int) -> int:
    """16 if d < 512; 12 if d >= 512; 24 if k > 16,384 (the k rule takes precedence, A14b)."""
    if k > 16_384:
        return 24
    return 16 if d < 512 else 12


def nominal_comparisons_per_point(nlist: int, nprobe: int, cell_size: int = DEFAULT_CELL_SIZE) -> int:
    """nlist + nprobe * cell_size  [O §04.1 table]."""
    return nlist + nprobe * cell_size


class CellIndex:
    def __init__(
        self,
        k: int,
        d: int,
        nlist: int | None = None,
        nprobe: int | None = None,
        cell_size: int = DEFAULT_CELL_SIZE,
        kmeans_iters: int = 10,
        seed: int = 0,
    ) -> None:
        self.k, self.d, self.cell_size = k, d, cell_size
        self.nlist = int(nlist) if nlist is not None else default_nlist(k, cell_size)
        self.nprobe = int(nprobe) if nprobe is not None else default_nprobe(k, d)
        if self.nlist > k:
            raise ValueError(f"nlist={self.nlist} > k={k}")
        self.kmeans_iters = kmeans_iters
        self._seed = seed
        self._n_builds = 0
        self.centres = torch.empty(0)
        self.cell_of = torch.zeros(k, dtype=torch.int64)
        self.child = torch.full((k,), -1, dtype=torch.int64)  # split pointers donor -> child (A17)
        self._dirty = True
        self.order = torch.arange(k)
        self.ptr = torch.zeros(self.nlist + 1, dtype=torch.int64)
        self.maintenance_ops = 0.0
        self.n_refresh = self.n_rebuild = 0

    # ------------------------------------------------------------------ build / maintain
    def _cell_kmeans(self, C: torch.Tensor) -> torch.Tensor:
        self._n_builds += 1
        res = Lloyd(empty_policy="split").fit(
            C.numpy(), self.nlist, Budget(iterations=self.kmeans_iters), seed=self._seed + 1009 * self._n_builds
        )
        self.maintenance_ops += float(self.k) * self.nlist * (self.kmeans_iters + 1)  # iterations + final filing
        return torch.from_numpy(res.centroids)

    def _file(self, C: torch.Tensor) -> None:
        self.cell_of, _ = assign(C, self.centres)
        self._dirty = True

    def build(self, C: torch.Tensor) -> None:
        self.centres = self._cell_kmeans(C)
        self._file(C)

    def rebuild(self, C: torch.Tensor) -> None:
        """Recompute the cell centres by k-means over the centroids [O §04.2]."""
        self.build(C)
        self.n_rebuild += 1

    def refresh(self, C: torch.Tensor) -> None:
        """Keep the cell centres; re-file every centroid under its nearest centre; no k-means iterations [O §04.2]."""
        self._file(C)
        self.maintenance_ops += float(self.k) * self.nlist
        self.n_refresh += 1

    def note_split(self, dead: int, donor: int, pointer: bool = True) -> None:
        """A revived slot starts in its donor's cell; record donor -> child pointer unless disabled (A17)."""
        self.cell_of[dead] = self.cell_of[donor]
        self.child[dead] = -1
        if pointer:
            self.child[donor] = dead
        self._dirty = True

    def _resort(self) -> None:
        self.order = torch.argsort(self.cell_of, stable=True)
        counts = torch.bincount(self.cell_of, minlength=self.nlist)
        self.ptr = torch.zeros(self.nlist + 1, dtype=torch.int64)
        self.ptr[1:] = torch.cumsum(counts, 0)
        self._dirty = False

    def cell_sizes(self) -> torch.Tensor:
        if self._dirty:
            self._resort()
        return self.ptr[1:] - self.ptr[:-1]

    # ------------------------------------------------------------------ search
    def search(self, xb: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Approximate nearest centroid for each row of xb. Returns (labels, d2 float32, op counts)."""
        if self._dirty:
            self._resort()
        B = xb.shape[0]
        sizes = self.ptr[1:] - self.ptr[:-1]
        nonempty = sizes > 0
        npb = min(self.nprobe, int(nonempty.sum()))

        Cs = C[self.order]
        cn = (Cs * Cs).sum(1)
        cc = self.centres
        ccn = (cc * cc).sum(1)
        s1 = torch.addmm(ccn[None, :], xb, cc.T, alpha=-2.0)  # level 1: B x nlist
        s1[:, ~nonempty] = float("inf")
        cell_ids = torch.topk(s1, npb, dim=1, largest=False).indices  # [B, npb]

        flat = cell_ids.reshape(-1)
        srt = torch.argsort(flat, stable=True)
        pts_sorted = srt // npb
        seg = torch.bincount(flat, minlength=self.nlist)
        seg_ptr = torch.zeros(self.nlist + 1, dtype=torch.int64)
        seg_ptr[1:] = torch.cumsum(seg, 0)

        best_v = torch.full((B,), float("inf"))
        best_i = torch.zeros(B, dtype=torch.int64)
        pair_ops = 0.0
        for c in torch.nonzero(seg > 0).flatten().tolist():  # level 2: one small dense product per cell
            s, e = int(self.ptr[c]), int(self.ptr[c + 1])
            pts = pts_sorted[int(seg_ptr[c]) : int(seg_ptr[c + 1])]
            sc = torch.addmm(cn[None, s:e], xb[pts], Cs[s:e].T, alpha=-2.0)
            v, ix = sc.min(dim=1)
            better = v < best_v[pts]
            pb = pts[better]
            best_v[pb] = v[better]
            best_i[pb] = self.order[s + ix[better]]
            pair_ops += float(pts.numel()) * (e - s)

        xx = (xb * xb).sum(1)
        d2 = (best_v + xx).clamp_min_(0.0)
        labels = best_i

        # split pointers: also evaluate the winner's child and grandchild [O §04.1 step d]
        ptr_ops = 0.0
        c1 = self.child[labels]
        m1 = torch.nonzero(c1 >= 0).flatten()
        if m1.numel():
            for cand in (c1[m1], self.child[c1[m1]]):
                ok = torch.nonzero(cand >= 0).flatten()
                if not ok.numel():
                    continue
                rows = m1[ok]
                diff = xb[rows] - C[cand[ok]]
                dist = (diff * diff).sum(1)
                better = dist < d2[rows]
                d2[rows[better]] = dist[better]
                labels[rows[better]] = cand[ok][better]
                ptr_ops += float(ok.numel())

        ops = {
            "cells": float(B) * self.nlist,
            "centroids": pair_ops,
            "pointer": ptr_ops,
            "nominal": float(B) * nominal_comparisons_per_point(self.nlist, self.nprobe, self.cell_size),
        }
        return labels, d2, ops
