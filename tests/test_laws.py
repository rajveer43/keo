import math

import numpy as np
import pytest

from keo.algos import laws

# Hand-check values: COMPUTED from the paper's printed coefficients (docs/SPEC.md §4.3), not measured results.
# (n, k, T, B_raw, B, alpha0, M)
TABLE = [
    (1_000_000, 16_384, 3, 57_333.8, 65_536, 0.36278, 45),
    (1_000_000, 16_384, 2, 41_283.5, 32_768, 0.20328, 60),
    (1_000_000, 1_024, 2, 28_790.0, 32_768, 0.72775, 60),
    (50_000, 512, 1, 1_074.9, 1_024, 0.18382, 48),
    (50_000, 512, 2, 1_884.5, 2_048, 0.31784, 48),
    (1_183_514, 4_096, 2, 39_985.4, 32_768, 0.35654, 72),
    (1_183_514, 16_384, 2, 47_881.7, 65_536, 0.35163, 36),
    (1_183_514, 32_768, 2, 52_396.6, 65_536, 0.25563, 36),
]


@pytest.mark.parametrize("n,k,T,B_raw,B,a0,M", TABLE)
def test_laws_match_hand_table(n, k, T, B_raw, B, a0, M):
    assert laws.batch_size_raw(n, k, T) == pytest.approx(B_raw, rel=1e-4)
    assert laws.batch_size(n, k, T) == B
    assert laws.alpha0(n, k, T, B) == pytest.approx(a0, abs=6e-6)
    assert laws.n_updates(n, B, T) == M


def test_law_reproduces_paper_figure7_knee_at_T25():
    # [E Fig. 7 caption]: knee 220,202 at T=25 (DEEP n=1e6, k=1024). Formula is within ~2 %.
    assert laws.batch_size_raw(10**6, 1024, 25) == pytest.approx(220_202, rel=0.02)


def test_large_n_divides_by_8_and_0_0044_equivalence():
    n, k, T = 10_000_000, 65_536, 3
    raw = laws.batch_size_raw(n, k, T)
    unscaled = laws.BATCH_C * k**laws.BATCH_K * n**laws.BATCH_N * T**laws.BATCH_T
    assert raw == pytest.approx(unscaled / 8)
    assert laws.BATCH_C / 8 == pytest.approx(0.0044, abs=1e-4)  # [E Table 2 third row]
    # boundary: exactly 2M is NOT large
    assert laws.batch_size_raw(2_000_000, 1000, 2) == pytest.approx(
        laws.BATCH_C * 1000**laws.BATCH_K * 2_000_000**laws.BATCH_N * 2**laws.BATCH_T
    )


def test_batch_is_power_of_two_and_capped_at_n():
    for n, k, T in [(5000, 64, 1.5), (123_457, 777, 3.3), (1_183_514, 4096, 12.8)]:
        B = laws.batch_size(n, k, T)
        assert B == 2 ** round(math.log2(B)) and 1 <= B <= n
    assert laws.batch_size(10, 5, 3) <= 10  # tiny n: capped


def test_alpha0_is_clipped_at_one():
    assert laws.alpha0(10_000, 2, 1.0, 10_000) == 1.0


def test_round_pow2_log_space():
    assert laws.round_pow2(46_000) == 32_768  # below log-midpoint 46,341
    assert laws.round_pow2(46_500) == 65_536
    assert laws.round_pow2(0.3) == 1


def test_schedule_endpoints_and_monotone():
    M, a0, eps = 60, 0.3, 0.1
    s = laws.alpha_schedule(M, a0, eps)
    assert s[0] == pytest.approx(a0)  # alpha(0) = alpha0
    assert laws.alpha_at(M, M, a0, eps) == pytest.approx(eps * a0)  # limit alpha(M) = eps*alpha0
    assert (np.diff(s) <= 1e-15).all()
    assert eps * a0 <= s[-1] < eps * a0 * 1.02


def test_n_updates_fractional_epochs_floor():
    assert laws.n_updates(1000, 100, 1.5) == 15
    assert laws.n_updates(1000, 300, 2.5) == 7  # floor(2.5*3)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        laws.batch_size(0, 10, 1)
    with pytest.raises(ValueError):
        laws.alpha0(100, 10, 0, 10)
