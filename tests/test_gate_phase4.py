"""PHASE 4 GATE: o-means vs e-means at equal epochs on the GloVe smoke subset (n=50k, k=512), same init per seed.

Smoke override: cell_size=8 (instead of 128) so that k=512 yields nlist=64 cells with nprobe=16 -> the search is truly
approximate (nominal 64 + 16*8 = 192 of 512 comparisons = 37.5 %). With the default cell size k=512 would give
nlist=16=nprobe, i.e. exact search, which would test nothing.

Gate: median over seeds of |rel. test MSE of o-means vs e-means| <= 3 % at every T in {1, 2, 3}, and the
missed-nearest rate is logged (finite). Everything else is printed for context.
Run with `pytest -s tests/test_gate_phase4.py`.
"""

import numpy as np
import pytest

from keo.algos import Budget, EMeans, Lloyd, OMeans
from keo.algos.init import random_row_init
from keo.config import load_dataset_config
from keo.data.download import dataset_file
from keo.data.prep import prepare, subset_train
from keo.metrics import mse

pytestmark = [pytest.mark.data, pytest.mark.slow]

N, K, SEEDS, EPOCHS, CELL = 50_000, 512, (0, 1, 2, 3, 4), (1, 2, 3), 8
TOL = 0.03


@pytest.fixture(scope="module")
def smoke_data():
    cfg = load_dataset_config("glove100")
    if not dataset_file(cfg).is_file():
        pytest.skip("run `keo download glove100` first")
    p = prepare(cfg)
    return subset_train(p.train, N, seed=0), p.test


def test_omeans_within_few_percent_of_emeans_and_probe_logged(smoke_data):
    train, test = smoke_data
    rows = {T: [] for T in EPOCHS}
    for seed in SEEDS:
        C0, _ = random_row_init(train, K, seed)
        ll = mse(test, Lloyd().fit(train, K, Budget(iterations=25), seed=seed, init_centroids=C0).centroids)
        for T in EPOCHS:
            e = EMeans().fit(train, K, Budget(epochs=T), seed=seed, init_centroids=C0)
            o = OMeans(cell_size=CELL).fit(train, K, Budget(epochs=T), seed=seed, init_centroids=C0)
            te_e, te_o = mse(test, e.centroids), mse(test, o.centroids)
            d = o.diagnostics
            rows[T].append(dict(
                o_vs_e=(te_o - te_e) / te_e, e_vs_l=(te_e - ll) / ll, o_vs_l=(te_o - ll) / ll,
                missed=d["missed_nearest_mean"], frac_actual=d["assign_fraction_of_exact_actual"],
                frac_nominal=d["assign_fraction_of_exact_nominal"],
                passes_o=o.op_counts["assign_passes"], passes_e=e.op_counts["assign_passes"],
                finish=d["finish"], nref=d["n_refresh"], nreb=d["n_rebuild"],
                actual_cs=d["cell_size_actual_visited_mean"], ptrs=d["n_split_pointers"],
            ))
    ex = rows[2][0]
    print(f"\nsmoke: n={N} k={K} d=100 | o-means nlist=64 nprobe=16 cell_size={CELL} (nominal 192/512 = 37.5% of exact)")
    print("median over 5 seeds; rel. TEST MSE (negative = better); 'vs L25' = vs Lloyd@25")
    print(" T | finish           | O vs E  | E vs L25 | O vs L25 | missed-nearest | assign ops/exact (actual|nominal) |"
          " passes E -> O (incl. maintenance)")
    for T in EPOCHS:
        r = rows[T]
        med = lambda key: float(np.median([x[key] for x in r]))  # noqa: E731
        print(f" {T} | {r[0]['finish']:<16} | {med('o_vs_e'):+7.3%} | {med('e_vs_l'):+8.3%} | {med('o_vs_l'):+8.3%} | "
              f"{med('missed'):13.3%} | {med('frac_actual'):.3f} | {med('frac_nominal'):.3f}"
              f"{'':>22}| {med('passes_e'):.2f} -> {med('passes_o'):.2f}")
    print(f"refresh/rebuild events (T=2, seed 0): {ex['nref']}/{ex['nreb']} ; split pointers at end: {ex['ptrs']} ; "
          f"mean ACTUAL visited cell size {ex['actual_cs']:.1f} vs nominal {CELL}")
    for T in EPOCHS:
        assert abs(np.median([x["o_vs_e"] for x in rows[T]])) <= TOL, f"T={T}"
        assert all(np.isfinite(x["missed"]) for x in rows[T])
