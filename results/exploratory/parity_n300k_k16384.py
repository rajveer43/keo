"""Exploratory (NOT a gate): reference e-means vs OFFICIAL faiss.contrib.e_means.NumpyEmeans at n=300k, k=16384 (n/k=18),
d=100, GloVe-100 normalised, same init rows, 3 seeds, T in {2, 3}. Reference backend (torch CPU) vs official (faiss CPU)."""
import time
import numpy as np
from keo.config import load_dataset_config
from keo.data.prep import prepare, subset_train
from keo.algos import Budget, EMeans
from keo.algos.adapters.faiss_official import OfficialEmeans
from keo.algos.init import random_row_init
from keo.metrics import mse

p = prepare(load_dataset_config("glove100"))
n, k = 300_000, 16_384
train, test = subset_train(p.train, n, 0), p.test
print(f"parity: n={n} k={k} d=100 same init, 3 seeds; rel. TEST/TRAIN MSE of the reference vs the official implementation", flush=True)
R = {}
t_all = time.time()
for seed in (0, 1, 2):
    C0, _ = random_row_init(train, k, seed)
    for T in (2, 3):
        off = OfficialEmeans().fit(train, k, Budget(epochs=T), seed=seed, init_centroids=C0)
        te, tr = mse(test, off.centroids), mse(train, off.centroids)
        for name, rec in (("official", "official"), ("paper", "paper")):
            r = EMeans(recipe=rec).fit(train, k, Budget(epochs=T), seed=seed, init_centroids=C0)
            R.setdefault((T, name, "test"), []).append((mse(test, r.centroids) - te) / te)
            R.setdefault((T, name, "train"), []).append((mse(train, r.centroids) - tr) / tr)
            if name == "official":
                R.setdefault((T, "recipe"), []).append((off.diagnostics["batch_size"], round(off.diagnostics["alpha0"], 4), r.diagnostics["batch_size"], round(r.diagnostics["alpha0"], 4), off.diagnostics["n_updates"], r.diagnostics["n_updates"], off.diagnostics["n_revived_total"], r.diagnostics["n_revived_total"]))
        print(f"  seed {seed} T={T} done ({time.time()-t_all:.0f}s)", flush=True)
print("\n=== medians over seeds [min, max]  (reference minus official, relative) ===")
for key, v in R.items():
    if key[1] == "recipe": print(f"  T={key[0]} (official B, alpha0, ref B, alpha0, official updates, ref updates, official revived, ref revived): {v[0]}"); continue
    a = np.array(v); print(f"  T={key[0]} ref[{key[1]} recipe] {key[2]:5s} MSE vs official: {np.median(a):+8.3%}  [{a.min():+7.3%}, {a.max():+7.3%}]")
print(f"total {time.time()-t_all:.0f}s")
