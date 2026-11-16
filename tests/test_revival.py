import numpy as np
import torch

from keo.algos.revival import BF16_EPS, dead_indices, revive_resample, revive_split


def _state(k=64, d=6, dead=(3, 17, 40), seed=0):
    rng = np.random.default_rng(seed)
    C = torch.from_numpy(rng.standard_normal((k, d)).astype(np.float32))
    N = torch.from_numpy(rng.uniform(8, 12, k).astype(np.float32))  # healthy: ~mean 10
    for j in dead:
        N[j] = 0.01  # << 0.1 * mean
    return C * N[:, None], N, C


def test_bf16_epsilon_value():
    assert BF16_EPS == 2.0**-7 == 0.0078125


def test_split_revival_mass_accounting_and_halving():
    S, N, C = _state()
    dead = dead_indices(N, 0.1).tolist()
    assert sorted(dead) == [3, 17, 40]
    N0, S0 = N.clone(), S.clone()
    old_dead_mass = float(N0[dead].sum())
    st = revive_split(S, N, 0.1, np.random.default_rng(1))
    assert (st.n_dead, st.n_revived, st.n_skipped) == (3, 3, 0)
    # total mass changes by exactly minus the dead clusters' old mass
    assert float(N.sum()) == np.float32(float(N0.sum()) - old_dead_mass) or abs(
        float(N.sum()) - (float(N0.sum()) - old_dead_mass)
    ) < 1e-3
    changed = (N != N0).nonzero().flatten().tolist()
    donors = [j for j in changed if j not in dead]
    assert len(donors) == 3 and len(set(donors)) == 3  # distinct donors (without replacement)
    for dj, j in zip(sorted(dead), donors):
        pass
    for j in donors:
        assert float(N[j]) == np.float32(float(N0[j]) / 2)
    # each dead slot's count equals half of some donor's old count
    halves = sorted(float(N0[j]) / 2 for j in donors)
    assert np.allclose(sorted(float(N[j]) for j in dead), halves, rtol=1e-6)


def test_split_children_are_alternating_one_pm_delta_perturbations():
    S, N, C = _state(dead=(5,))
    N0, C0 = N.clone(), C.clone()
    revive_split(S, N, 0.1, np.random.default_rng(2))
    donor = [j for j in range(len(N)) if j != 5 and N[j] != N0[j]][0]
    c_dead, c_donor = S[5] / N[5], S[donor] / N[donor]
    up = torch.ones(C.shape[1]); down = torch.ones(C.shape[1])
    up[0::2], up[1::2] = 1 + BF16_EPS, 1 - BF16_EPS
    down[0::2], down[1::2] = 1 - BF16_EPS, 1 + BF16_EPS
    assert torch.allclose(c_dead, C0[donor] * up, rtol=1e-5)
    assert torch.allclose(c_donor, C0[donor] * down, rtol=1e-5)


def test_after_revival_no_cluster_below_threshold_when_donors_are_healthy():
    # healthy donors have N >= 8 => N/2 >= 4 >> 0.1*mean ~ 1  ->  nobody is dead afterwards
    S, N, _ = _state()
    revive_split(S, N, 0.1, np.random.default_rng(3))
    assert dead_indices(N, 0.1).numel() == 0


def test_revival_no_dead_is_noop():
    S, N, _ = _state(dead=())
    S0, N0 = S.clone(), N.clone()
    st = revive_split(S, N, 0.1, np.random.default_rng(0))
    assert st.n_dead == 0 and torch.equal(S, S0) and torch.equal(N, N0)


def test_revival_donor_pool_exhausted_is_skipped_and_counted():
    k, d = 6, 3
    N = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # a single live cluster
    S = torch.ones(k, d) * N[:, None]
    st = revive_split(S, N, 0.1, np.random.default_rng(0))
    assert st.n_dead == 5 and st.n_revived == 1 and st.n_skipped == 4


def test_donor_probability_proportional_to_count():
    k, d = 4, 2
    hits = np.zeros(k)
    for s in range(400):
        N = torch.tensor([100.0, 1.0, 1.0, 0.0])  # cluster 3 dead; donors 0 (p~.98), 1, 2
        S = torch.ones(k, d) * N[:, None]
        revive_split(S, N, 0.1, np.random.default_rng(s))
        hits[(N == 50.0).nonzero().flatten().tolist() or [0]] += 1
    assert hits[0] > 0.9 * 400


def test_resample_refills_with_batch_vectors():
    S, N, _ = _state(dead=(2, 9))
    xb = torch.randn(32, S.shape[1])
    mean_n = float(N.mean())
    st = revive_resample(S, N, 0.1, xb, np.random.default_rng(0))
    assert st.n_revived == 2
    for j in (2, 9):
        assert float(N[j]) == np.float32(mean_n) or abs(float(N[j]) - mean_n) < 1e-4
        c = S[j] / N[j]
        assert any(torch.allclose(c, row, atol=1e-5) for row in xb)
