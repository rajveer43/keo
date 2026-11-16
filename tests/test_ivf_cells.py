import numpy as np
import pytest
import torch

from keo.algos.ivf_cells import (
    CellIndex,
    default_nlist,
    default_nprobe,
    nominal_comparisons_per_point,
)
from keo.metrics import assign


def _clustered(k=256, d=12, groups=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((groups, d)).astype(np.float32) * 6
    C = np.vstack([c + 0.5 * rng.standard_normal((k // groups, d)).astype(np.float32) for c in centers])
    X = np.vstack([c + 0.8 * rng.standard_normal((200, d)).astype(np.float32) for c in centers])
    return torch.from_numpy(C), torch.from_numpy(X)


# ------------------------------------------------------------------ defaults / arithmetic [O §04 table]
@pytest.mark.parametrize(
    "k,d,nlist,nprobe,comparisons,share",
    [
        (8_192, 512, 64, 12, 1_600, 0.195),
        (16_384, 100, 128, 16, 2_176, 0.133),  # the case named in the Phase-4 gate: 128 + 16*128
        (16_384, 512, 128, 12, 1_664, 0.102),
        (65_536, 100, 512, 24, 3_584, 0.055),
    ],
)
def test_defaults_and_comparison_arithmetic_match_docs_table(k, d, nlist, nprobe, comparisons, share):
    assert default_nlist(k) == nlist
    assert default_nprobe(k, d) == nprobe
    c = nominal_comparisons_per_point(nlist, nprobe)
    assert c == comparisons
    assert c / k == pytest.approx(share, abs=0.001)


def test_nlist_rounds_up_to_multiple_of_16():
    for k in (100, 512, 2049, 4096, 20_000, 65_536):
        nl = default_nlist(k)
        assert nl % 16 == 0 and nl >= -(-k // 128)
    assert default_nlist(512) == 16 and default_nlist(4096) == 32 and default_nlist(32_768) == 256


def test_nprobe_rule_k_over_16384_takes_precedence():
    assert default_nprobe(16_384, 100) == 16 and default_nprobe(16_385, 100) == 24
    assert default_nprobe(40_000, 1024) == 24 and default_nprobe(1000, 1024) == 12


# ------------------------------------------------------------------ search correctness
def test_exhaustive_probe_equals_exact_search():
    C, X = _clustered()
    idx = CellIndex(256, 12, nlist=16, nprobe=16, cell_size=16, seed=1)
    idx.build(C)
    lab, d2, _ = idx.search(X, C)
    lab_e, d2_e = assign(X, C)
    assert (lab == lab_e).float().mean() > 0.9999
    torch.testing.assert_close(d2, d2_e, rtol=1e-4, atol=1e-4)


def test_approximate_search_never_beats_exact_and_labels_valid():
    C, X = _clustered()
    idx = CellIndex(256, 12, nlist=16, nprobe=2, cell_size=16, seed=1)
    idx.build(C)
    lab, d2, ops = idx.search(X, C)
    _, d2_e = assign(X, C)
    assert ((lab >= 0) & (lab < 256)).all()
    # approximate distance is never smaller than the exact one, up to fp32 cancellation in ||x||^2 - 2x.c + ||c||^2,
    # whose error scales with ||x||^2 (observed ~1e-4 absolute at ||x||^2 ~ 300 with identical labels)
    tol = 2e-6 * (1.0 + (X * X).sum(1))
    assert (d2 >= d2_e - tol).all()
    assert (lab == assign(X, C)[0]).float().mean() > 0.8  # clustered data: probing 2 of 16 cells recovers most
    assert ops["centroids"] < X.shape[0] * 256  # fewer comparisons than brute force


def test_op_counts_equal_independent_recount():
    C, X = _clustered()
    idx = CellIndex(256, 12, nlist=16, nprobe=3, cell_size=16, seed=2)
    idx.build(C)
    _, _, ops = idx.search(X, C)
    sizes = idx.cell_sizes().double()
    s1 = (X @ idx.centres.T) * -2 + (idx.centres**2).sum(1)[None, :]
    s1[:, sizes == 0] = float("inf")
    cells = torch.topk(s1, 3, dim=1, largest=False).indices
    assert ops["cells"] == X.shape[0] * 16
    assert ops["centroids"] == float(sizes[cells].sum())
    assert ops["nominal"] == X.shape[0] * (16 + 3 * 16)


def test_empty_cells_are_skipped_and_every_point_gets_a_label():
    C, X = _clustered(k=64, groups=8)
    idx = CellIndex(64, 12, nlist=32, nprobe=8, cell_size=2, seed=3)  # far more cells than natural groups
    idx.build(C)
    idx.cell_of[:] = torch.randint(0, 5, (64,))  # force many empty cells
    idx._dirty = True
    lab, d2, _ = idx.search(X, C)
    assert (lab >= 0).all() and torch.isfinite(d2).all()
    assert int((idx.cell_sizes() == 0).sum()) >= 27


# ------------------------------------------------------------------ maintenance invariants
def test_refresh_files_each_centroid_in_exactly_one_nearest_cell():
    C, _ = _clustered()
    idx = CellIndex(256, 12, nlist=16, nprobe=4, cell_size=16, seed=4)
    idx.build(C)
    C2 = C + 0.7 * torch.randn_like(C)  # centroids drifted
    centres_before = idx.centres.clone()
    ops_before = idx.maintenance_ops
    idx.refresh(C2)
    torch.testing.assert_close(idx.centres, centres_before)  # refresh keeps the cell centres
    assert idx.maintenance_ops == ops_before + 256 * 16  # k * nlist, no k-means iterations
    sizes = idx.cell_sizes()
    assert int(sizes.sum()) == 256
    assert sorted(idx.order.tolist()) == list(range(256))  # a permutation: every centroid exactly once
    assert (idx.cell_of == assign(C2, idx.centres)[0]).all()  # filed under the nearest centre


def test_rebuild_recomputes_centres_and_counts_kmeans_cost():
    C, _ = _clustered()
    idx = CellIndex(256, 12, nlist=16, nprobe=4, cell_size=16, kmeans_iters=5, seed=5)
    idx.build(C)
    first = idx.maintenance_ops
    assert first == 256 * 16 * (5 + 1)
    idx.rebuild(C + 2.0 * torch.randn_like(C))
    assert idx.maintenance_ops == 2 * first and idx.n_rebuild == 1


def test_note_split_files_child_in_donor_cell_and_sets_pointer():
    C, _ = _clustered()
    idx = CellIndex(256, 12, nlist=16, nprobe=4, cell_size=16, seed=6)
    idx.build(C)
    idx.note_split(dead=10, donor=200)
    assert idx.cell_of[10] == idx.cell_of[200] and idx.child[200] == 10 and idx.child[10] == -1
    assert int(idx.cell_sizes().sum()) == 256


def _crafted(k_extra=False):
    # centroids on a line; cell 0 = {0, 1}, cell 1 = {2, 3, 4}; centroids 3 and 4 sit near centroid 1 but are filed in
    # the far cell (stale index: they were split off / drifted after the last refresh).
    C = torch.tensor([[0.0], [0.1], [10.0], [0.11], [0.115]])
    idx = CellIndex(5, 1, nlist=2, nprobe=1, cell_size=2, seed=0)
    idx.centres = torch.tensor([[0.0], [10.0]])
    idx.cell_of = torch.tensor([0, 0, 1, 1, 1])
    idx._dirty = True
    return idx, C


def test_without_pointers_a_stale_filing_misses_the_nearest_centroid():
    idx, C = _crafted()
    lab, _, _ = idx.search(torch.tensor([[0.116]]), C)
    assert int(lab[0]) == 1  # probes only cell 0 -> best visible is centroid 1 (true nearest is 4)


def test_child_pointer_recovers_nearest_centroid():
    idx, C = _crafted()
    idx.child[1] = 3
    lab, d2, ops = idx.search(torch.tensor([[0.116]]), C)
    assert int(lab[0]) == 3 and ops["pointer"] == 1


def test_grandchild_pointer_chain_is_followed():
    idx, C = _crafted()
    idx.child[1] = 3
    idx.child[3] = 4
    lab, d2, ops = idx.search(torch.tensor([[0.116]]), C)
    assert int(lab[0]) == 4 and ops["pointer"] == 2
    assert float(d2[0]) == pytest.approx((0.116 - 0.115) ** 2, abs=1e-7)


def test_nlist_larger_than_k_rejected():
    with pytest.raises(ValueError):
        CellIndex(10, 3, nlist=11)
