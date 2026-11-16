"""Dead-cluster revival for EMA k-means. Source: [E §3.1, App. H.1-H.2].

A cluster is dead when N_j < tau * mean(N) (tau = 0.1). `split` (default, the paper's strategy) rebuilds a dead
cluster by splitting a donor drawn WITHOUT replacement with probability proportional to N:

    c_dead  = c_donor * [1+d, 1-d, 1+d, ...]
    c_donor = c_donor * [1-d, 1+d, 1-d, ...]      d = BF16 machine epsilon = 2^-7
    N_dead = N_donor = N_donor_old / 2 ;  S = c * N

Total mass therefore changes by exactly -N_dead_old (the dead cluster's old, small count is discarded); this is
asserted in tests (A37). `resample` (ablation) refills the slot with a random vector from the current batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

BF16_EPS = 2.0**-7


@dataclass
class RevivalStats:
    n_dead: int = 0
    n_revived: int = 0
    n_skipped: int = 0  # dead clusters left untouched because the donor pool was exhausted (A6)
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (dead_slot, donor) for every split performed


def dead_indices(N: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.nonzero(N < tau * N.mean(), as_tuple=False).flatten()


def _patterns(d: int, delta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    up = torch.ones(d, dtype=dtype)
    down = torch.ones(d, dtype=dtype)
    up[0::2], up[1::2] = 1.0 + delta, 1.0 - delta
    down[0::2], down[1::2] = 1.0 - delta, 1.0 + delta
    return up, down


def revive_split(
    S: torch.Tensor,
    N: torch.Tensor,
    tau: float,
    rng: np.random.Generator,
    delta: float = BF16_EPS,
) -> RevivalStats:
    """In-place split-based revival on EMA state (S: [k,d], N: [k])."""
    stats = RevivalStats()
    dead = dead_indices(N, tau)
    stats.n_dead = int(dead.numel())
    if stats.n_dead == 0:
        return stats

    k = N.shape[0]
    is_dead = torch.zeros(k, dtype=torch.bool)
    is_dead[dead] = True
    pool = torch.nonzero(~is_dead & (N > 0), as_tuple=False).flatten().numpy()
    m = min(stats.n_dead, len(pool))
    stats.n_skipped = stats.n_dead - m
    if m == 0:
        return stats

    # deadest first, so that if donors run out the least-populated clusters are the ones revived
    order = torch.argsort(N[dead])
    dead = dead[order][:m].numpy()
    w = N[torch.from_numpy(pool)].double().numpy()
    donors = rng.choice(pool, size=m, replace=False, p=w / w.sum())

    up, down = _patterns(S.shape[1], delta, S.dtype)
    for dj, j in zip(dead, donors):
        n_j = N[j].clone()
        c = S[j] / n_j
        half = n_j / 2
        S[dj] = c * up * half
        S[j] = c * down * half
        N[dj] = half
        N[j] = half
        stats.pairs.append((int(dj), int(j)))
    stats.n_revived = m
    return stats


def revive_resample(
    S: torch.Tensor,
    N: torch.Tensor,
    tau: float,
    xb: torch.Tensor,
    rng: np.random.Generator,
) -> RevivalStats:
    """Ablation: refill each dead slot with a random vector of the current batch (A38: N := mean(N))."""
    stats = RevivalStats()
    dead = dead_indices(N, tau)
    stats.n_dead = int(dead.numel())
    if stats.n_dead == 0:
        return stats
    new_n = N.mean().clone()
    pick = rng.integers(0, xb.shape[0], size=stats.n_dead)
    S[dead] = xb[torch.from_numpy(pick)] * new_n
    N[dead] = new_n
    stats.n_revived = stats.n_dead
    return stats
