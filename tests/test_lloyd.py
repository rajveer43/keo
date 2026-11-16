import numpy as np
import pytest

from keo.algos import Budget, Lloyd
from keo.algos.adapters import faiss_cpu
from keo.algos.init import kmeanspp_init, random_row_init
from keo.metrics import assign, cluster_stats, mse


def blobs(n_per=300, k=4, d=6, sep=20.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)).astype(np.float32) * sep
    X = np.vstack([c + rng.standard_normal((n_per, d)).astype(np.float32) for c in centers])
    return X.astype(np.float32), centers


def test_budget_validation():
    with pytest.raises(ValueError):
        Budget()
    with pytest.raises(ValueError):
        Budget(epochs=1.0, iterations=1)
    with pytest.raises(ValueError):
        Budget(iterations=0)
    with pytest.raises(ValueError):
        Budget(epochs=-1)


def test_init_random_rows_distinct_deterministic_and_shared():
    X, _ = blobs()
    C1, i1 = random_row_init(X, 10, seed=3)
    C2, i2 = random_row_init(X, 10, seed=3)
    np.testing.assert_array_equal(C1, C2)
    assert len(set(i1.tolist())) == 10
    np.testing.assert_array_equal(C1, X[i1])
    assert not np.array_equal(C1, random_row_init(X, 10, seed=4)[0])


def test_kmeanspp_init_rows_from_data_and_deterministic():
    X, _ = blobs()
    C1, i1 = kmeanspp_init(X, 4, seed=1)
    C2, _ = kmeanspp_init(X, 4, seed=1)
    np.testing.assert_array_equal(C1, C2)
    np.testing.assert_array_equal(C1, X[i1])


def test_lloyd_recovers_well_separated_blobs():
    X, _ = blobs(k=4, d=6)
    r = Lloyd(init="kmeanspp").fit(X, 4, Budget(iterations=10), seed=0)
    # unit-variance noise in 6 dims -> MSE ~ 6 when each blob has its own centroid
    assert 5.0 < mse(X, r.centroids) < 7.0
    assert cluster_stats(assign(X, r.centroids)[0], 4)["empty_clusters"] == 0


def test_lloyd_objective_monotone_with_keep_policy():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((2000, 8)).astype(np.float32)
    r = Lloyd(empty_policy="keep").fit(X, 32, Budget(iterations=15), seed=2)
    obj = np.asarray(r.diagnostics["train_mse_per_iter"])
    assert (np.diff(obj) <= 1e-6 * obj[:-1]).all()


def test_lloyd_deterministic_and_seed_sensitive():
    X, _ = blobs()
    a = Lloyd().fit(X, 8, Budget(iterations=5), seed=7).centroids
    b = Lloyd().fit(X, 8, Budget(iterations=5), seed=7).centroids
    c = Lloyd().fit(X, 8, Budget(iterations=5), seed=8).centroids
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_checkpoint_snapshot_equals_independent_run():
    X, _ = blobs()
    full = Lloyd().fit(X, 8, Budget(iterations=6), seed=0, checkpoints=(2, 4))
    short = Lloyd().fit(X, 8, Budget(iterations=4), seed=0)
    np.testing.assert_array_equal(full.diagnostics["snapshots"][4][0], short.centroids)
    t2, t4 = full.diagnostics["snapshots"][2][1], full.diagnostics["snapshots"][4][1]
    assert 0 < t2 < t4


def _init_with_dead_slot(X, k, seed):
    C0, _ = random_row_init(X, k, seed)
    C0[-1] = 1e4  # nobody will ever be assigned here
    return C0


def test_empty_policy_keep_vs_split():
    X, _ = blobs(n_per=200, k=4, d=6)
    k = 6
    C0 = _init_with_dead_slot(X, k, 0)
    keep = Lloyd(empty_policy="keep").fit(X, k, Budget(iterations=3), seed=0, init_centroids=C0)
    split = Lloyd(empty_policy="split").fit(X, k, Budget(iterations=3), seed=0, init_centroids=C0)
    # keep: the dead centroid stays exactly where it was
    np.testing.assert_array_equal(keep.centroids[-1], C0[-1])
    assert keep.diagnostics["n_split_total"] == 0
    # split: the slot is moved next to a donor and gets repopulated
    assert split.diagnostics["n_split_total"] >= 1
    assert not np.allclose(split.centroids[-1], C0[-1])
    assert np.abs(split.centroids[-1]).max() < 1e3


def test_split_perturbation_is_faiss_style_eps_1_over_1024():
    # one iteration, one dead slot: dead slot and donor must be a (1+eps, 1-eps) alternating pair
    X, _ = blobs(n_per=200, k=4, d=8)
    C0 = _init_with_dead_slot(X, 5, 1)
    r = Lloyd(empty_policy="split").fit(X, 5, Budget(iterations=1), seed=1, init_centroids=C0)
    C = r.centroids
    found = False
    for j in range(4):
        ratio = C[4] / C[j]
        dev = np.abs(ratio - 1.0)
        if np.all(dev > 0.0015) and np.all(dev < 0.0025):  # (1+e)/(1-e)-1 ~ 2e = 0.00195
            found = True
    assert found


def test_lloyd_rejects_noninteger_epochs_and_bad_options():
    X, _ = blobs()
    with pytest.raises(ValueError):
        Lloyd().fit(X, 4, Budget(epochs=2.5), seed=0)
    with pytest.raises(ValueError):
        Lloyd(init="nope")
    with pytest.raises(ValueError):
        Lloyd(empty_policy="nope")


def test_op_counts():
    X, _ = blobs(n_per=100, k=4)
    r = Lloyd().fit(X, 4, Budget(iterations=5), seed=0)
    assert r.op_counts["assign_passes"] == 5
    assert r.op_counts["distance_computations"] == X.shape[0] * 4 * 5


def test_faiss_adapter_runs_and_decreases_objective():
    ok, why = faiss_cpu.is_available()
    assert ok, why
    X, _ = blobs(n_per=500, k=6, d=8, sep=3.0)
    r = faiss_cpu.FaissLloyd().fit(X, 12, Budget(iterations=8), seed=0)
    assert r.centroids.shape == (12, 8) and r.centroids.dtype == np.float32
    obj = np.asarray(r.diagnostics["train_mse_per_iter"])
    assert len(obj) == 8 and obj[-1] <= obj[0]
