import numpy as np
import pytest
import torch

from keo.algos import Budget, EMeans, OMeans
from keo.algos.emeans import RecencyAccumulator
from keo.metrics import mse


def blobs(n_per=500, k=8, d=8, sep=12.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)).astype(np.float32) * sep
    X = np.vstack([c + rng.standard_normal((n_per, d)).astype(np.float32) for c in centers])
    return X.astype(np.float32)


KW = dict(batch_size=200, alpha0=0.25)  # fixed so E and O share exactly the same schedule


def test_exhaustive_probe_matches_emeans_for_T_ge_2():
    """With nprobe == nlist the two-level search is exact, so o-means must follow the e-means trajectory."""
    X = blobs()
    k = 64
    e = EMeans(**KW).fit(X, k, Budget(epochs=3), seed=0)
    o = OMeans(nlist=16, nprobe=16, cell_size=4, **KW).fit(X, k, Budget(epochs=3), seed=0)
    assert o.diagnostics["finish"] == "last_epoch_mean"
    assert abs(mse(X, o.centroids) - mse(X, e.centroids)) / mse(X, e.centroids) < 1e-3
    np.testing.assert_allclose(o.diagnostics["batch_mse"][0], e.diagnostics["batch_mse"][0], rtol=1e-5)  # same init


def test_same_seed_same_init_and_deterministic():
    X = blobs()
    f = lambda s: OMeans(nlist=16, nprobe=4, cell_size=4, **KW).fit(X, 64, Budget(epochs=2), seed=s)  # noqa: E731
    a, b, c = f(3), f(3), f(4)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    assert not np.array_equal(a.centroids, c.centroids)
    assert a.diagnostics["init"] == "random"


@pytest.mark.parametrize(
    "T,expected",
    [(1.0, "recency"), (1.5, "recency"), (1.99, "recency"), (2.0, "last_epoch_mean"), (3.0, "last_epoch_mean")],
)
def test_finishing_switches_at_two_epochs(T, expected):
    X = blobs()
    r = OMeans(nlist=16, nprobe=8, cell_size=4, **KW).fit(X, 64, Budget(epochs=T), seed=0)
    assert r.diagnostics["finish"] == expected


def test_finish_none_disables_recency_and_threshold_is_configurable():
    X = blobs()
    r = OMeans(nlist=16, nprobe=8, cell_size=4, finalize="none", **KW).fit(X, 64, Budget(epochs=1), seed=0)
    assert r.diagnostics["finish"] == "none"
    r = OMeans(nlist=16, nprobe=8, cell_size=4, recency_below_epochs=0.5, **KW).fit(X, 64, Budget(epochs=1), seed=0)
    assert r.diagnostics["finish"] == "last_epoch_mean"


def test_maintenance_schedule_counts():
    """n=4000, B=100 -> 40 batches/epoch; T=3 -> 120 updates.
    refresh after updates 5,10,...,40 (8x); rebuild every half epoch (20) after epoch 1: 60, 80, 100, 120 (4x)."""
    X = blobs(n_per=500, k=8)  # n = 4000
    o = OMeans(nlist=16, nprobe=4, cell_size=4, batch_size=100, alpha0=0.2).fit(X, 64, Budget(epochs=3), seed=0)
    d = o.diagnostics
    assert d["batches_per_epoch"] == 40 and d["n_updates"] == 120
    assert d["refresh_batches"] == [5, 10, 15, 20, 25, 30, 35, 40]
    assert d["rebuild_batches"] == [60, 80, 100, 120]
    assert (d["n_refresh"], d["n_rebuild"]) == (8, 4)
    k, nlist, iters = 64, 16, 10
    expected_maint = k * nlist * (iters + 1) * (1 + 4) + k * nlist * 8
    assert o.op_counts["maintenance_distance_computations"] == expected_maint
    assert o.op_counts["distance_computations"] == pytest.approx(
        o.op_counts["assign_distance_computations"] + expected_maint
    )


def test_short_first_epoch_gets_one_forced_refresh():
    X = blobs(n_per=50, k=8)  # n = 400, B=100 -> 4 batches/epoch (< 5)
    o = OMeans(nlist=16, nprobe=4, cell_size=4, batch_size=100, alpha0=0.2).fit(X, 64, Budget(epochs=1), seed=0)
    assert o.diagnostics["refresh_batches"] == [4] and o.diagnostics["n_rebuild"] == 0


def test_missed_nearest_probe_logged_per_batch_and_bounded():
    X = blobs()
    o = OMeans(nlist=16, nprobe=2, cell_size=4, **KW).fit(X, 64, Budget(epochs=2), seed=0)
    m = o.diagnostics["missed_nearest_per_batch"]
    assert len(m) == o.diagnostics["n_updates"] and all(0.0 <= v <= 1.0 for v in m)
    ex = OMeans(nlist=16, nprobe=16, cell_size=4, **KW).fit(X, 64, Budget(epochs=2), seed=0)
    assert ex.diagnostics["missed_nearest_mean"] < 0.01  # exhaustive probe: ~nothing missed (fp ties only)


def test_probe_cost_is_excluded_from_op_counts():
    X = blobs()
    a = OMeans(nlist=16, nprobe=4, cell_size=4, probe_frac=0.01, **KW).fit(X, 64, Budget(epochs=2), seed=0)
    b = OMeans(nlist=16, nprobe=4, cell_size=4, probe_frac=0.5, **KW).fit(X, 64, Budget(epochs=2), seed=0)
    assert a.op_counts["assign_distance_computations"] == b.op_counts["assign_distance_computations"]


def test_fewer_distance_computations_than_emeans_when_probing_a_subset():
    X = blobs(n_per=1000, k=8)
    e = EMeans(**KW).fit(X, 128, Budget(epochs=2), seed=0)
    o = OMeans(nlist=16, nprobe=2, cell_size=8, **KW).fit(X, 128, Budget(epochs=2), seed=0)
    assert o.op_counts["distance_computations"] < e.op_counts["distance_computations"]
    assert o.diagnostics["assign_fraction_of_exact_actual"] < 0.5
    assert e.op_counts["distance_computations"] == e.op_counts["exact_equivalent_distance_computations"]


def test_split_pointers_recorded_when_revival_fires():
    X = blobs(n_per=400, k=4, d=6)
    k = 32
    from keo.algos.init import random_row_init

    C0, _ = random_row_init(X, k, 0)
    C0[16:] = 1e3
    o = OMeans(nlist=16, nprobe=8, cell_size=2, batch_size=400, alpha0=0.4, finalize="none").fit(
        X, k, Budget(epochs=10), seed=0, init_centroids=C0
    )
    assert o.diagnostics["n_revived_total"] > 0
    assert o.diagnostics["n_split_pointers"] > 0
    assert np.isfinite(o.centroids).all()


def test_assign_exact_matches_metrics():
    X = blobs()
    o = OMeans(nlist=16, nprobe=8, cell_size=4, **KW).fit(X, 64, Budget(epochs=2), seed=0)
    lab, d2 = OMeans.assign_exact(X, o.centroids)
    assert lab.shape == (X.shape[0],) and d2.shape == (X.shape[0],)
    assert float(np.mean(d2.astype(np.float64))) == pytest.approx(mse(X, o.centroids), rel=1e-4)


def test_requires_epochs_budget():
    with pytest.raises(ValueError):
        OMeans().fit(blobs(), 8, Budget(iterations=3), seed=0)


# ------------------------------------------------------------------ recency accumulator [O §04.4, A18]
def test_recency_accumulator_decay_and_weights():
    acc = RecencyAccumulator(k=2, d=1, batches_per_epoch=4, tau_epochs=1.0)
    lab = torch.tensor([0, 0, 1])
    x = torch.tensor([[1.0], [3.0], [10.0]])
    acc.update(lab, x)
    acc.update(torch.tensor([0]), torch.tensor([[5.0]]))
    dec = np.exp(-0.25)
    assert float(acc.W[0]) == pytest.approx(2 * dec + 1) and float(acc.A[0, 0]) == pytest.approx(4 * dec + 5)
    c = acc.centroids(fallback=torch.zeros(2, 1))
    assert float(c[0, 0]) == pytest.approx((4 * dec + 5) / (2 * dec + 1), rel=1e-5)
    assert float(c[1, 0]) == pytest.approx(10.0)  # exp-weighted mean of a single value is that value


def test_recency_split_halves_donor_sums_into_child_and_unseen_clusters_fall_back():
    acc = RecencyAccumulator(k=4, d=2, batches_per_epoch=5, tau_epochs=1.0)
    acc.A[1], acc.W[1] = torch.tensor([6.0, 8.0], dtype=torch.float64), 4.0
    acc.A[3], acc.W[3] = torch.tensor([99.0, 99.0], dtype=torch.float64), 9.0  # dead slot's stale mass is replaced
    acc.split([(3, 1)])
    assert torch.equal(acc.A[1], acc.A[3]) and float(acc.W[1]) == float(acc.W[3]) == 2.0
    assert acc.A[1].tolist() == [3.0, 4.0]
    fb = torch.full((4, 2), -1.0)
    c = acc.centroids(fb)
    assert c[0].tolist() == [-1.0, -1.0]  # never seen -> falls back to the EMA centroid
    assert c[1].tolist() == [1.5, 2.0]


# ------------------------------------------------------------------ ablation switches [O §04.2]
def test_static_index_ablation_never_maintains():
    X = blobs(n_per=500, k=8)
    o = OMeans(nlist=16, nprobe=4, cell_size=4, maintain=False, batch_size=100, alpha0=0.2).fit(
        X, 64, Budget(epochs=3), seed=0
    )
    d = o.diagnostics
    assert (d["n_refresh"], d["n_rebuild"], d["n_split_pointers"]) == (0, 0, 0)
    assert o.op_counts["maintenance_distance_computations"] == 64 * 16 * 11  # the initial build only
    assert np.isfinite(o.centroids).all()


def test_pointers_off_keeps_maintenance_but_no_pointers():
    X = blobs(n_per=400, k=4, d=6)
    from keo.algos.init import random_row_init

    C0, _ = random_row_init(X, 32, 0)
    C0[16:] = 1e3
    kw = dict(nlist=16, nprobe=8, cell_size=2, batch_size=400, alpha0=0.4, finalize="none")
    on = OMeans(**kw).fit(X, 32, Budget(epochs=10), seed=0, init_centroids=C0)
    off = OMeans(pointers=False, **kw).fit(X, 32, Budget(epochs=10), seed=0, init_centroids=C0)
    assert on.diagnostics["n_split_pointers"] > 0 and off.diagnostics["n_split_pointers"] == 0
    assert off.diagnostics["n_refresh"] == on.diagnostics["n_refresh"] > 0
