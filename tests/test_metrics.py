import numpy as np
import torch

from keo.metrics import assign, cluster_stats, mse


def _brute(X, C):
    d2 = ((X[:, None, :].astype(np.float64) - C[None, :, :].astype(np.float64)) ** 2).sum(-1)
    return d2.argmin(1), d2.min(1)


def test_assign_matches_bruteforce_across_chunk_sizes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 12)).astype(np.float32)
    C = rng.standard_normal((37, 12)).astype(np.float32)
    lab_ref, d2_ref = _brute(X, C)
    for chunk_elems in (1, 37, 1000, 1 << 20):  # 1 -> one row per chunk
        lab, d2 = assign(X, C, chunk_elems=chunk_elems)
        np.testing.assert_array_equal(lab.numpy(), lab_ref)
        np.testing.assert_allclose(d2.numpy(), d2_ref, rtol=1e-4, atol=1e-4)


def test_assign_min_distance_never_negative():
    X = np.ones((10, 5), dtype=np.float32) * 1e3
    _, d2 = assign(X, X[:3].copy())
    assert (d2 >= 0).all()


def test_mse_float64_matches_numpy():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 7)).astype(np.float32)
    C = rng.standard_normal((9, 7)).astype(np.float32)
    _, d2_ref = _brute(X, C)
    assert abs(mse(X, C) - d2_ref.mean()) < 1e-9 * max(1.0, d2_ref.mean()) + 1e-7


def test_mse_is_sum_over_coordinates_not_mean():
    X = np.zeros((4, 10), dtype=np.float32)
    C = np.ones((1, 10), dtype=np.float32)
    assert mse(X, C) == 10.0  # [E Eq.1]: ||x-c||^2 summed over d


def test_cluster_stats_known_labels():
    lab = torch.tensor([0] * 6 + [1] * 2 + [2] * 2 + [3] * 0)  # k=4 -> cluster 3 empty
    s = cluster_stats(lab, 4)
    assert s["empty_clusters"] == 1
    assert s["dead_clusters"] == 4  # every cluster has < 10 points
    # sizes 6,2,2,0 ; n=10 ; k*sum(s^2)/n^2 = 4*44/100
    assert abs(s["imbalance_factor"] - 1.76) < 1e-12
    assert abs(s["max_over_mean"] - 6 / 2.5) < 1e-12


def test_cluster_stats_balanced_is_one():
    lab = torch.arange(100) % 5
    s = cluster_stats(lab, 5)
    assert abs(s["imbalance_factor"] - 1.0) < 1e-12 and s["dead_clusters"] == 0
