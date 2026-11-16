"""PHASE 5 GATE: reference e-means vs the OFFICIAL faiss.contrib.e_means.NumpyEmeans on the GloVe smoke subset.

n=50,000, k=512, d=100, same init rows per seed, 5 seeds. Gate: median over seeds of |rel. TEST MSE difference| between the
reference with the `official` recipe (constants shipped in faiss) and the official implementation is <= 0.5 % at T=3.
Also printed (for the gap documentation): reference with the `paper` recipe, T in {1,2,3}, train MSE, and Lloyd@25.
Run with `pytest -s tests/test_gate_phase5.py`.
"""

import numpy as np
import pytest

from keo.algos import Budget, EMeans, Lloyd
from keo.algos.adapters.faiss_official import OfficialEmeans, emeans_available
from keo.algos.init import random_row_init
from keo.config import load_dataset_config
from keo.data.download import dataset_file
from keo.data.prep import prepare, subset_train
from keo.metrics import mse

pytestmark = [pytest.mark.data, pytest.mark.slow]

N, K, SEEDS, EPOCHS = 50_000, 512, (0, 1, 2, 3, 4), (1, 2, 3)
TOL = 0.005


@pytest.fixture(scope="module")
def smoke_data():
    cfg = load_dataset_config("glove100")
    if not dataset_file(cfg).is_file():
        pytest.skip("run `keo download glove100` first")
    ok, why = emeans_available()
    if not ok:
        pytest.skip(why)
    p = prepare(cfg)
    return subset_train(p.train, N, seed=0), p.test


def test_reference_matches_official_emeans(smoke_data):
    train, test = smoke_data
    rows = {T: [] for T in EPOCHS}
    for seed in SEEDS:
        C0, _ = random_row_init(train, K, seed)
        l25 = mse(test, Lloyd().fit(train, K, Budget(iterations=25), seed=seed, init_centroids=C0).centroids)
        for T in EPOCHS:
            off = OfficialEmeans().fit(train, K, Budget(epochs=T), seed=seed, init_centroids=C0)
            ref_o = EMeans(recipe="official").fit(train, K, Budget(epochs=T), seed=seed, init_centroids=C0)
            ref_p = EMeans(recipe="paper").fit(train, K, Budget(epochs=T), seed=seed, init_centroids=C0)
            te_off = mse(test, off.centroids)
            rows[T].append(dict(
                ref_off=(mse(test, ref_o.centroids) - te_off) / te_off,
                ref_paper=(mse(test, ref_p.centroids) - te_off) / te_off,
                ref_off_train=(mse(train, ref_o.centroids) - mse(train, off.centroids)) / mse(train, off.centroids),
                off_vs_l25=(te_off - l25) / l25,
                B_off=off.diagnostics["batch_size"], B_ref=ref_o.diagnostics["batch_size"], B_paper=ref_p.diagnostics["batch_size"],
                a0_off=off.diagnostics["alpha0"], a0_ref=ref_o.diagnostics["alpha0"],
                steps_off=off.diagnostics["n_updates"], steps_ref=ref_o.diagnostics["n_updates"],
                passes_off=off.op_counts["assign_passes"], passes_ref=ref_o.op_counts["assign_passes"],
            ))
    r0 = rows[3][0]
    print(f"\nsmoke n={N} k={K} d=100. recipe check (T=3): batch official={r0['B_off']} ref(official)={r0['B_ref']} "
          f"ref(paper)={r0['B_paper']} | alpha0 official={r0['a0_off']:.5f} ref(official)={r0['a0_ref']:.5f}")
    print(f"updates official={r0['steps_off']} (ceil(n/B)/pass, remainder kept) vs ref={r0['steps_ref']} (floor); "
          f"passes official={r0['passes_off']:.2f} ref={r0['passes_ref']:.2f}")
    print("rel. TEST MSE of the REFERENCE vs the OFFICIAL implementation (same init), median [min, max] over 5 seeds")
    print(" T | ref[official recipe]           | ref[paper recipe]              | ref[official] TRAIN | official vs Lloyd@25")
    for T in EPOCHS:
        f = lambda key: np.array([x[key] for x in rows[T]])  # noqa: E731
        a, b, c, e = f("ref_off"), f("ref_paper"), f("ref_off_train"), f("off_vs_l25")
        print(f" {T} | {np.median(a):+7.3%} [{a.min():+7.3%}, {a.max():+7.3%}] | {np.median(b):+7.3%} [{b.min():+7.3%}, {b.max():+7.3%}]"
              f" | {np.median(c):+7.3%}          | {np.median(e):+7.3%}")
    assert abs(np.median([x["ref_off"] for x in rows[3]])) <= TOL
