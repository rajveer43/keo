import time, numpy as np
from keo.config import load_dataset_config
from keo.data.prep import prepare, subset_train
from keo.algos import Budget, Lloyd, EMeans
from keo.algos.init import random_row_init
from keo.metrics import mse, assign

p = prepare(load_dataset_config("glove100"))
n, k = 300_000, 16_384
train = subset_train(p.train, n, 0); test = p.test
print(f"exploratory: n={n} k={k} (n/k={n/k:.1f}) d=100, 3 seeds", flush=True)
variants = {
  "recipe":        dict(),
  "no_cosine":     dict(cosine=False),
  "revival=off":   dict(revival="off"),
  "revival=resamp":dict(revival="resample"),
  "no_finalize":   dict(finalize="none"),
}
rows = {}
t_all = time.time()
for seed in (0, 1, 2):
    C0, _ = random_row_init(train, k, seed)
    t = time.time(); ll = Lloyd().fit(train, k, Budget(iterations=25), seed=seed, init_centroids=C0, checkpoints=(5, 10, 25))
    ref = mse(test, ll.centroids)
    print(f"seed {seed}: Lloyd@25 test={ref:.5f} ({time.time()-t:.0f}s)  splits={ll.diagnostics['n_split_total']}", flush=True)
    for it in (5, 10):
        rows.setdefault(f"lloyd@{it}", []).append((mse(test, ll.diagnostics['snapshots'][it][0]) - ref) / ref)
    for T in (1, 1.5, 2, 3, 5):
        r = EMeans().fit(train, k, Budget(epochs=T), seed=seed, init_centroids=C0)
        rows.setdefault(f"E T={T}", []).append((mse(test, r.centroids) - ref) / ref)
        if T == 3:
            d = r.diagnostics
            rows.setdefault("info", []).append((d["batch_size"], round(d["alpha0"], 4), d["n_updates"], d["n_revived_total"], round(r.timings["fit"], 1)))
    for name, kw in variants.items():
        if name == "recipe": continue
        r = EMeans(**kw).fit(train, k, Budget(epochs=3), seed=seed, init_centroids=C0)
        rows.setdefault(f"abl@3 {name}", []).append((mse(test, r.centroids) - ref) / ref)
print("\nrel. TEST MSE vs Lloyd@25 (median [min,max] over seeds; negative = better than Lloyd@25)")
for key, v in rows.items():
    if key == "info": continue
    a = np.array(v); print(f"  {key:20s} {np.median(a):+8.3%}  [{a.min():+7.3%}, {a.max():+7.3%}]")
print("E@3 (B, alpha0, updates, revived_total, fit_s) per seed:", rows["info"])
print(f"total {time.time()-t_all:.0f}s")
