"""Exploratory (NOT a gate): o-means vs e-means near the paper's regime. n=300k GloVe-100 (normalised), k=16384, d=100,
default o-means settings (cell_size 128 -> nlist 128, nprobe 16). 3 seeds. Reference backend, CPU.
Run: uv run python -I results/exploratory/omeans_n300k_k16384.py > results/exploratory/omeans_n300k_k16384.txt
"""
import time
import numpy as np
from keo.config import load_dataset_config
from keo.data.prep import prepare, subset_train
from keo.algos import Budget, Lloyd, EMeans, OMeans
from keo.algos.init import random_row_init
from keo.metrics import mse

p = prepare(load_dataset_config("glove100"))
n, k = 300_000, 16_384
train, test = subset_train(p.train, n, 0), p.test
print(f"exploratory: n={n} k={k} (n/k={n/k:.1f}) d=100, 3 seeds, o-means defaults (nlist=128, nprobe=16, cell_size=128)", flush=True)
R = {}
def add(key, v): R.setdefault(key, []).append(v)
t_all = time.time()
for seed in (0, 1, 2):
    C0, _ = random_row_init(train, k, seed)
    ll = mse(test, Lloyd().fit(train, k, Budget(iterations=25), seed=seed, init_centroids=C0).centroids)
    print(f"seed {seed}: Lloyd@25 test={ll:.5f}", flush=True)
    def run_o(T, **kw):
        t = time.time(); o = OMeans(**kw).fit(train, k, Budget(epochs=T), seed=seed, init_centroids=C0)
        return o, mse(test, o.centroids), time.time() - t
    for T in (1, 1.5, 2, 3):
        t = time.time(); e = EMeans().fit(train, k, Budget(epochs=T), seed=seed, init_centroids=C0); te_e = mse(test, e.centroids); t_e = time.time() - t
        o, te_o, t_o = run_o(T); d = o.diagnostics
        add(f"T={T} O vs E", (te_o - te_e) / te_e); add(f"T={T} E vs L25", (te_e - ll) / ll); add(f"T={T} O vs L25", (te_o - ll) / ll)
        add(f"T={T} missed", d["missed_nearest_mean"]); add(f"T={T} frac_actual", d["assign_fraction_of_exact_actual"])
        add(f"T={T} frac_nominal", d["assign_fraction_of_exact_nominal"]); add(f"T={T} cs_actual", d["cell_size_actual_visited_mean"])
        add(f"T={T} passes_E", e.op_counts["assign_passes"]); add(f"T={T} passes_O", o.op_counts["assign_passes"])
        add(f"T={T} maint_share", o.op_counts["maintenance_distance_computations"] / o.op_counts["distance_computations"])
        add(f"T={T} sec_E", t_e); add(f"T={T} sec_O", t_o)
        add(f"T={T} finish", d["finish"]); add(f"T={T} refresh/rebuild", (d["n_refresh"], d["n_rebuild"]))
        add(f"T={T} missed_first_batches", float(np.mean(d["missed_nearest_per_batch"][:5]))); add(f"T={T} missed_last_batches", float(np.mean(d["missed_nearest_per_batch"][-5:])))
        print(f"  seed {seed} T={T}: O vs E {(te_o-te_e)/te_e:+.3%}  missed {d['missed_nearest_mean']:.2%}  ops frac {d['assign_fraction_of_exact_actual']:.3f}", flush=True)
    # C9: maintained vs static index (T=2)
    for name, kw in {"maintained": {}, "no-pointers": dict(pointers=False), "static-index": dict(maintain=False)}.items():
        o, te_o, _ = run_o(2, **kw)
        add(f"C9 T=2 {name}: missed", o.diagnostics["missed_nearest_mean"]); add(f"C9 T=2 {name}: test vs L25", (te_o - ll) / ll)
        add(f"C9 T=2 {name}: missed (first 5 batches)", float(np.mean(o.diagnostics["missed_nearest_per_batch"][:5])))
    # C10: recency finish vs last-epoch-mean
    for T, kw, name in [(1, {}, "recency (default)"), (1, dict(recency_below_epochs=0.0), "last-epoch-mean"),
                        (1.5, {}, "recency (default)"), (1.5, dict(recency_below_epochs=0.0), "last-epoch-mean"),
                        (2, {}, "last-epoch-mean (default)"), (2, dict(recency_below_epochs=3.0), "recency (forced)")]:
        o, te_o, _ = run_o(T, **kw); add(f"C10 T={T} {name}: test vs L25", (te_o - ll) / ll)
print("\n=== medians over seeds [min, max] ===")
for key, v in R.items():
    if isinstance(v[0], str): print(f"  {key:44s} {v[0]}"); continue
    if isinstance(v[0], tuple): print(f"  {key:44s} {v[0]}"); continue
    a = np.array(v)
    fmt = (lambda x: f"{x:+8.3%}") if ("vs" in key or "missed" in key or "frac" in key or "share" in key) else (lambda x: f"{x:9.3f}")
    print(f"  {key:44s} {fmt(np.median(a))}  [{fmt(a.min())}, {fmt(a.max())}]")
print(f"total {time.time()-t_all:.0f}s")
