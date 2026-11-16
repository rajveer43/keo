"""e-means scaling laws and step schedule. Pure functions (no state, no torch).

Source: [E §3.3, Table 2, Eq. 5-7]. Laws were fitted on DEEP only; coefficients are the paper's printed (rounded) values (A3).

    alpha0 = min( 0.824 * k^-0.46 * B^0.90 * n^-0.45 * T^-0.11 , 1 )                       (Eq. 6)
    B      = round_to_nearest_pow2( 0.035 * k^0.13 * n^0.88 * T^0.81 ), capped at n        (Eq. 7)
             for n > 2e6 the law value is divided by 8 first (0.0044 = 0.035/8)
    alpha(t) = alpha0 * ( eps + (1-eps) * 0.5 * (1 + cos(pi * t / M)) ),  eps = 0.1         (Eq. 5)
"""

from __future__ import annotations

import math

import numpy as np

# Step law coefficients (c, k, B, n, T) [E Table 2, row 1]
ALPHA_C, ALPHA_K, ALPHA_B, ALPHA_N, ALPHA_T = 0.824, -0.46, 0.90, -0.45, -0.11
# Batch law coefficients (c, k, n, T) [E Table 2, rows 2-3]
BATCH_C, BATCH_K, BATCH_N, BATCH_T = 0.035, 0.13, 0.88, 0.81
LARGE_N_THRESHOLD = 2_000_000
LARGE_N_DIVISOR = 8.0
COSINE_EPS = 0.1


def round_pow2(x: float) -> int:
    """Round to the nearest power of two in log2 space (A11)."""
    if x <= 1:
        return 1
    return int(2 ** round(math.log2(x)))


def batch_size_raw(n: int, k: int, T: float) -> float:
    """Unrounded batch-law value (Eq. 7), including the /8 rule for n > 2M."""
    v = BATCH_C * k**BATCH_K * n**BATCH_N * T**BATCH_T
    if n > LARGE_N_THRESHOLD:
        v /= LARGE_N_DIVISOR
    return v


def batch_size(n: int, k: int, T: float) -> int:
    """Batch size B from the law: nearest power of two, clamped to [1, n] (A12)."""
    if n < 1 or k < 1 or T <= 0:
        raise ValueError("need n>=1, k>=1, T>0")
    return int(min(max(round_pow2(batch_size_raw(n, k, T)), 1), n))


def alpha0(n: int, k: int, T: float, B: int) -> float:
    """Peak step alpha0 from the law (Eq. 6), clipped to <= 1."""
    if n < 1 or k < 1 or T <= 0 or B < 1:
        raise ValueError("need n>=1, k>=1, T>0, B>=1")
    return min(ALPHA_C * k**ALPHA_K * B**ALPHA_B * n**ALPHA_N * T**ALPHA_T, 1.0)


def n_updates(n: int, B: int, T: float) -> int:
    """M = floor(T * floor(n/B)) (A1)."""
    return int(math.floor(T * (n // B)))


def alpha_at(t: int, M: int, a0: float, eps: float = COSINE_EPS) -> float:
    """Cosine-annealed step at update t in {0..M-1} (Eq. 5): alpha(0)=a0, alpha(M)->eps*a0."""
    return a0 * (eps + (1.0 - eps) * 0.5 * (1.0 + math.cos(math.pi * t / M)))


def alpha_schedule(M: int, a0: float, eps: float = COSINE_EPS) -> np.ndarray:
    return np.array([alpha_at(t, M, a0, eps) for t in range(M)], dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------
# "official" recipe: the constants and rules SHIPPED in faiss.contrib.e_means (faiss-cpu 1.16.0, base.py), which its
# own comments say "differ from those reported in the paper's Table 2". Verified against the shipped code by
# tests/test_adapters.py::test_official_recipe_matches_shipped_code (runs the shipped `_resolve_recipe` in a worker).
# ---------------------------------------------------------------------------------------------------------------
OFFICIAL_BATCH = {  # (C_B, k exp, n exp, T exp)
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),  # = throughput / 8
}
OFFICIAL_ALPHA = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)  # (C, k exp, B exp, n exp, T exp)
OFFICIAL_ALPHA_CLAMP = (1e-4, 1.0)
OFFICIAL_AUTO_QUALITY_MAX_D = 16  # batch_law="auto": quality law for d <= 16, throughput above
OFFICIAL_NO_REVIVAL_MAX_D = 2  # revival is switched off for d <= 2
OFFICIAL_N_INIT = 2.0**-16  # initial EMA mass (paper: 1e-10)
OFFICIAL_SPLIT_DELTA = 1e-4  # donor-split perturbation (paper: BF16 eps = 2^-7)


def official_batch_law(d: int, batch_law: str = "auto") -> str:
    if batch_law != "auto":
        return batch_law
    return "quality" if d <= OFFICIAL_AUTO_QUALITY_MAX_D else "throughput"


def round_pow2_ratio(value: float) -> int:
    """The shipped rounding: nearest power of two by ratio (== rounding in log2 space)."""
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def official_recipe(n: int, k: int, T: float, d: int, batch_law: str = "auto") -> tuple[int, float, str]:
    """(batch size B, alpha0, resolved batch law) exactly as the shipped `_resolve_recipe` computes them."""
    law = official_batch_law(d, batch_law)
    c, bk, bn, bt = OFFICIAL_BATCH[law]
    B = round_pow2_ratio(c * k**bk * n**bn * T**bt)
    B = min(B, round_pow2_ratio(n / 2)) if law == "throughput" else min(B, n // 2)
    B = max(1, min(B, n))
    ca, ak, ab, an, at = OFFICIAL_ALPHA
    a0 = min(max(ca * k**ak * B**ab * n**an * T**at, OFFICIAL_ALPHA_CLAMP[0]), OFFICIAL_ALPHA_CLAMP[1])
    return B, float(a0), law
