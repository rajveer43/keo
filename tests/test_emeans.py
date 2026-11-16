import numpy as np
import pytest

from keo.algos import Budget, EMeans, Lloyd
from keo.algos import laws
from keo.algos.init import random_row_init
from keo.metrics import assign, mse


def blobs(n_per=500, k=6, d=8, sep=15.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)).astype(np.float32) * sep
    X = np.vstack([c + rng.standard_normal((n_per, d)).astype(np.float32) for c in centers])
    return X.astype(np.float32)


def test_invariant_full_batch_alpha1_equals_lloyd_mean():
    """SPEC §4.7(1): B=n and alpha=1 for a single update => S/N is the Lloyd mean of the assigned points."""
    X = blobs()
    n, k = X.shape[0], 12
    C0, _ = random_row_init(X, k, 0)
    r = EMeans(batch_size=n, alpha0=1.0, finalize="none", revival="off").fit(
        X, k, Budget(epochs=1), seed=0, init_centroids=C0
    )
    assert r.diagnostics["n_updates"] == 1 and r.diagnostics["batch_size"] == n
    ll = Lloyd(empty_policy="keep").fit(X, k, Budget(iterations=1), seed=0, init_centroids=C0)
    np.testing.assert_allclose(r.centroids, ll.centroids, rtol=1e-5, atol=1e-5)


def test_alpha1_does_not_produce_nan_for_unused_clusters():
    # alpha=1 wipes N,S of unused clusters (0/0); the fallback keeps the last valid centroid (A39)
    X = blobs(k=3)
    C0, _ = random_row_init(X, 3, 0)
    C0 = np.vstack([C0, np.full((1, X.shape[1]), 1e4, dtype=np.float32)])  # slot nobody will use
    r = EMeans(batch_size=X.shape[0], alpha0=1.0, finalize="none", revival="off").fit(
        X, 4, Budget(epochs=1), seed=0, init_centroids=C0
    )
    assert np.isfinite(r.centroids).all()
    np.testing.assert_allclose(r.centroids[3], C0[3])


def test_law_defaults_are_used_and_reported():
    X = blobs()
    n, k, T = X.shape[0], 16, 2.0
    r = EMeans(recipe="paper").fit(X, k, Budget(epochs=T), seed=0)
    B = laws.batch_size(n, k, T)
    assert r.diagnostics["batch_size"] == B
    assert r.diagnostics["alpha0"] == pytest.approx(laws.alpha0(n, k, T, B))
    assert r.diagnostics["n_updates"] == laws.n_updates(n, B, T)


def test_overrides_for_law_transfer_sweep():
    X = blobs()
    r = EMeans(batch_size=256, alpha0=0.05).fit(X, 8, Budget(epochs=2), seed=0)
    assert r.diagnostics["batch_size"] == 256 and r.diagnostics["alpha0"] == 0.05


def test_deterministic_same_seed_and_seed_sensitive():
    X = blobs()
    f = lambda s: EMeans().fit(X, 12, Budget(epochs=2), seed=s).centroids  # noqa: E731
    np.testing.assert_array_equal(f(3), f(3))
    assert not np.array_equal(f(3), f(4))


def test_same_seed_same_init_as_lloyd():
    X = blobs()
    e = EMeans(finalize="none").fit(X, 9, Budget(epochs=1), seed=5)
    assert e.diagnostics["init"] == "random"
    C0, _ = random_row_init(X, 9, 5)
    r2 = EMeans(finalize="none").fit(X, 9, Budget(epochs=1), seed=5, init_centroids=C0)
    np.testing.assert_array_equal(e.centroids, r2.centroids)  # explicit shared init == seed-derived init


def test_finalize_last_epoch_mean_never_increases_train_mse_vs_none():
    X = blobs(n_per=800, k=8, d=6, sep=3.0)
    base = dict(batch_size=256, alpha0=0.2, revival="off")
    none = EMeans(finalize="none", **base).fit(X, 24, Budget(epochs=3), seed=1)
    lem = EMeans(finalize="last_epoch_mean", **base).fit(X, 24, Budget(epochs=3), seed=1)
    # same trajectory => centroids_before_finalize identical
    np.testing.assert_array_equal(lem.diagnostics["centroids_before_finalize"], none.centroids)
    assert mse(X, lem.centroids) <= mse(X, none.centroids) * (1 + 1e-6)


def test_last_epoch_labels_recorded():
    X = blobs()
    r = EMeans(batch_size=512, alpha0=0.3).fit(X, 10, Budget(epochs=2), seed=0)
    lab = r.train_labels
    nb = X.shape[0] // 512
    assert (lab >= 0).sum() == nb * 512  # last epoch visits nb*B distinct points
    assert lab.max() < 10


def test_op_counts_one_pass_per_epoch_when_B_divides_n():
    X = blobs(n_per=500, k=4)  # n = 2000
    r = EMeans(batch_size=500, alpha0=0.2).fit(X, 8, Budget(epochs=3), seed=0)
    assert r.op_counts["distance_computations"] == 3 * 4 * 500 * 8
    assert r.op_counts["assign_passes"] == pytest.approx(3.0)


def test_revival_split_fires_and_prevents_dead_clusters():
    X = blobs(n_per=400, k=4, d=6)
    k = 16
    C0, _ = random_row_init(X, k, 0)
    C0[8:] = 1e3  # half of the centroids start far away: they get no points
    kw = dict(batch_size=400, alpha0=0.4, finalize="none")
    off = EMeans(revival="off", **kw).fit(X, k, Budget(epochs=10), seed=0, init_centroids=C0)
    on = EMeans(revival="split", **kw).fit(X, k, Budget(epochs=10), seed=0, init_centroids=C0)
    assert on.diagnostics["n_revived_total"] > 0 and off.diagnostics["n_revived_total"] == 0
    # mechanism: the 8 far-away slots never receive points without revival, and all get repopulated with it
    empties = lambda r: int((np.bincount(assign(X, r.centroids)[0].numpy(), minlength=k) == 0).sum())  # noqa: E731
    assert empties(off) == 8 and empties(on) == 0
    # and the extra live centroids lower the error (4 blobs: 16 centroids cannot halve it, so no big margin)
    assert mse(X, on.centroids) < mse(X, off.centroids)


def test_ablation_flags_run_and_validate():
    X = blobs()
    for kw in (dict(cosine=False), dict(revival="resample"), dict(revival="off"),
               dict(finalize="exact_pass"), dict(finalize="none"), dict(init="kmeanspp")):
        r = EMeans(**kw).fit(X, 10, Budget(epochs=2), seed=0)
        assert np.isfinite(r.centroids).all() and r.centroids.shape == (10, X.shape[1])
    with pytest.raises(ValueError):
        EMeans(revival="x")
    with pytest.raises(ValueError):
        EMeans(finalize="x")
    with pytest.raises(ValueError):
        EMeans().fit(X, 4, Budget(iterations=3), seed=0)


def test_exact_pass_finalize_costs_one_extra_pass():
    X = blobs(n_per=500, k=4)
    a = EMeans(batch_size=500, alpha0=0.2, finalize="none").fit(X, 8, Budget(epochs=2), seed=0)
    b = EMeans(batch_size=500, alpha0=0.2, finalize="exact_pass").fit(X, 8, Budget(epochs=2), seed=0)
    assert b.op_counts["assign_passes"] == pytest.approx(a.op_counts["assign_passes"] + 1.0)


def test_learns_well_separated_blobs():
    X = blobs(n_per=600, k=5, d=6, sep=25.0)
    r = EMeans().fit(X, 5, Budget(epochs=5), seed=0)
    assert mse(X, r.centroids) < 8.0  # unit noise in 6 dims -> ~6 at the optimum
    lab, _ = assign(X, r.centroids)
    assert len(set(lab.tolist())) == 5


def test_default_recipe_is_official_and_reports_its_law():
    X = blobs()
    n, k, d = X.shape[0], 16, X.shape[1]
    r = EMeans().fit(X, k, Budget(epochs=2.0), seed=0)
    B, a0, law = laws.official_recipe(n, k, 2.0, d)
    assert r.diagnostics["recipe"] == "official"
    assert (r.diagnostics["batch_size"], r.diagnostics["batch_law"]) == (B, law)
    assert r.diagnostics["alpha0"] == pytest.approx(a0)
