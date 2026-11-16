"""PHASE 2 GATE: reference Lloyd vs Faiss-CPU Lloyd on the GloVe smoke subset, same init, 25 iterations.

Smoke subset: n=50,000 (fixed-seed subset of normalised train), k=512, test split = 5,000 held-out queries.
Gate: median across seeds of |relative test-MSE difference| <= 0.1 %.
(Per-seed values are printed; run with `pytest -s tests/test_gate_phase2.py`.)
"""

import numpy as np
import pytest

from keo.algos import Budget, Lloyd
from keo.algos.adapters.faiss_cpu import FaissLloyd
from keo.algos.init import random_row_init
from keo.config import load_dataset_config
from keo.data.download import dataset_file
from keo.data.prep import prepare, subset_train
from keo.metrics import mse

pytestmark = [pytest.mark.data, pytest.mark.slow]

N, K, ITERS, SEEDS = 50_000, 512, 25, (0, 1, 2, 3, 4)
TOL = 0.001


@pytest.fixture(scope="module")
def smoke_data():
    cfg = load_dataset_config("glove100")
    if not dataset_file(cfg).is_file():
        pytest.skip("run `keo download glove100` first")
    p = prepare(cfg)
    return subset_train(p.train, N, seed=0), p.test


def test_reference_lloyd_matches_faiss(smoke_data):
    train, test = smoke_data
    rows = []
    for seed in SEEDS:
        C0, _ = random_row_init(train, K, seed)
        ref = Lloyd().fit(train, K, Budget(iterations=ITERS), seed=seed, init_centroids=C0)
        off = FaissLloyd().fit(train, K, Budget(iterations=ITERS), seed=seed, init_centroids=C0)
        te_r, te_o = mse(test, ref.centroids), mse(test, off.centroids)
        tr_r, tr_o = mse(train, ref.centroids), mse(train, off.centroids)
        rows.append((seed, te_r, te_o, (te_r - te_o) / te_o, (tr_r - tr_o) / tr_o,
                     ref.diagnostics["n_split_total"], ref.timings["fit"], off.timings["fit"]))
    print("\nseed  test_ref   test_faiss  rel_test   rel_train  splits  t_ref(s) t_faiss(s)")
    for r in rows:
        print(f"{r[0]:>4}  {r[1]:.6f}  {r[2]:.6f}  {r[3]:+.5%}  {r[4]:+.5%}  {r[5]:>5}  {r[6]:7.2f}  {r[7]:7.2f}")
    rel = np.array([abs(r[3]) for r in rows])
    print(f"median |rel test diff| = {np.median(rel):.5%} ; max = {rel.max():.5%}  (gate {TOL:.1%})")
    assert np.median(rel) <= TOL
