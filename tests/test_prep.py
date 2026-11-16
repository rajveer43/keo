import numpy as np

from keo.data.prep import (
    count_internal_duplicates,
    dedup_eval_against_train,
    l2_normalize,
    norm_stats,
    split_val_test,
    subset_train,
)


def test_dedup_removes_train_overlap_and_internal_dups():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((50, 8)).astype(np.float32)
    fresh = rng.standard_normal((6, 8)).astype(np.float32)
    eval_ = np.vstack([train[3:5], fresh, fresh[:2]])  # 2 in train, 6 fresh, 2 internal dups
    keep, rep = dedup_eval_against_train(train, eval_)
    assert rep["eval_in"] == 10
    assert rep["eval_removed_present_in_train"] == 2
    assert rep["eval_removed_internal_duplicates"] == 2
    assert rep["eval_kept"] == 6 == len(keep)
    np.testing.assert_array_equal(eval_[keep], fresh)


def test_dedup_no_overlap_keeps_all():
    rng = np.random.default_rng(1)
    train = rng.standard_normal((20, 4)).astype(np.float32)
    eval_ = rng.standard_normal((5, 4)).astype(np.float32)
    keep, rep = dedup_eval_against_train(train, eval_)
    assert len(keep) == 5 and rep["eval_removed_present_in_train"] == 0


def test_internal_duplicates_count():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert count_internal_duplicates(np.vstack([a, a[:1]])) == 1


def test_l2_normalize_unit_norm_and_zero_rows_flagged():
    X = np.array([[3, 4], [0, 0], [1, 0]], dtype=np.float32)
    Xn, mask = l2_normalize(X)
    np.testing.assert_array_equal(mask, [True, False, True])
    np.testing.assert_allclose(np.linalg.norm(Xn[mask], axis=1), 1.0, atol=1e-6)
    assert Xn.dtype == np.float32


def test_norm_stats_detects_unit_norm():
    X = np.eye(5, dtype=np.float32)
    s = norm_stats(X)
    assert s["frac_within_1e-3_of_1"] == 1.0 and s["n_zero"] == 0


def test_split_is_partition_and_deterministic():
    v1, t1 = split_val_test(101, 0.5, seed=7)
    v2, t2 = split_val_test(101, 0.5, seed=7)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(t1, t2)
    assert len(set(v1) & set(t1)) == 0
    assert len(v1) + len(t1) == 101
    v3, _ = split_val_test(101, 0.5, seed=8)
    assert not np.array_equal(v1, v3)


def test_subset_train_deterministic_and_none_means_all():
    X = np.arange(200, dtype=np.float32).reshape(100, 2)
    assert subset_train(X, None, 0) is X
    a, b = subset_train(X, 30, 3), subset_train(X, 30, 3)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (30, 2)
    assert subset_train(X, 1000, 0) is X
