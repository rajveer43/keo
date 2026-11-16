"""PHASE 3 GATE: e-means at ~3 epochs vs Lloyd@25 on the GloVe smoke subset (n=50k, k=512), same init per seed.

Gate: median over seeds of the relative TEST MSE of e-means@3 vs Lloyd@25 is <= +0.1 % ("beats or matches").
All other epoch budgets are printed for context only. Run with `pytest -s tests/test_gate_phase3.py`.
"""

import numpy as np
import pytest

from keo.algos import Budget, EMeans, Lloyd
from keo.algos.init import random_row_init
from keo.config import load_dataset_config
from keo.data.download import dataset_file
from keo.data.prep import prepare, subset_train
from keo.metrics import mse

pytestmark = [pytest.mark.data, pytest.mark.slow]

N, K, SEEDS = 50_000, 512, (0, 1, 2, 3, 4)
EPOCHS = (1, 2, 3, 5)
TOL = 0.001


@pytest.fixture(scope="module")
def smoke_data():
    cfg = load_dataset_config("glove100")
    if not dataset_file(cfg).is_file():
        pytest.skip("run `keo download glove100` first")
    p = prepare(cfg)
    return subset_train(p.train, N, seed=0), p.test


def test_emeans_3_epochs_matches_or_beats_lloyd25(smoke_data):
    train, test = smoke_data
    rel = {T: [] for T in EPOCHS}
    rel_tr = {T: [] for T in EPOCHS}
    info = None
    for seed in SEEDS:
        C0, _ = random_row_init(train, K, seed)
        ll = Lloyd().fit(train, K, Budget(iterations=25), seed=seed, init_centroids=C0)
        ref_te, ref_tr = mse(test, ll.centroids), mse(train, ll.centroids)
        for T in EPOCHS:
            r = EMeans().fit(train, K, Budget(epochs=T), seed=seed, init_centroids=C0)
            rel[T].append((mse(test, r.centroids) - ref_te) / ref_te)
            rel_tr[T].append((mse(train, r.centroids) - ref_tr) / ref_tr)
            if T == 3 and info is None:
                info = {k: r.diagnostics[k] for k in ("batch_size", "alpha0", "n_updates", "n_revived_total")}
                info["passes"] = r.op_counts["assign_passes"]
    print(f"\nsmoke: n={N} k={K} d=100 | e-means@3 law settings: {info}")
    print("rel. MSE vs Lloyd@25 (negative = e-means better), median [min, max] over 5 seeds")
    print(" epochs |   test MSE                      |   train MSE")
    for T in EPOCHS:
        a, b = np.array(rel[T]), np.array(rel_tr[T])
        print(f"   {T:>4} | {np.median(a):+8.3%} [{a.min():+7.3%}, {a.max():+7.3%}] | "
              f"{np.median(b):+8.3%} [{b.min():+7.3%}, {b.max():+7.3%}]")
    assert np.median(rel[3]) <= TOL
