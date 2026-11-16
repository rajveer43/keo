"""Adapter availability, isolation and the official e-means (the only official implementation runnable on CPU here)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from keo.algos import Budget, EMeans, laws
from keo.algos.adapters import _worker_client as wc
from keo.algos.adapters import faiss_cpu, faiss_official, flash_kmeans
from keo.algos.adapters._worker_client import AdapterUnavailable
from keo.algos.adapters.registry import adapter_status, get_adapter
from keo.algos.init import random_row_init
from keo.metrics import mse


def blobs(n_per=600, k=8, d=12, sep=10.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)).astype(np.float32) * sep
    return np.vstack([c + rng.standard_normal((n_per, d)).astype(np.float32) for c in centers]).astype(np.float32)


# ------------------------------------------------------------------ availability (this machine: CPU wheel, no CUDA)
def test_registry_never_raises_and_reports_every_adapter():
    st = {s.key: s for s in adapter_status()}
    assert set(st) == {"faiss_lloyd_cpu", "faiss_lloyd_gpu", "flash_kmeans", "official_emeans_cpu", "official_omeans"}
    assert all(isinstance(s.reason, str) and s.reason for s in st.values())


def test_cpu_adapters_available_and_marked_verified():
    st = {s.key: s for s in adapter_status()}
    for key in ("faiss_lloyd_cpu", "official_emeans_cpu"):
        assert st[key].available and st[key].verified, st[key]


def test_gpu_adapter_not_fooled_by_faiss_cpu_stub():
    """Regression: faiss-cpu's get_num_gpus() returns 1 (stub); that must NOT make the GPU adapter 'available'."""
    ok, why = faiss_cpu.is_available(gpu=True)
    assert not ok and "no GPU" in why
    assert wc.probe()["num_gpus"] == 0
    with pytest.raises(AdapterUnavailable):
        faiss_cpu.FaissLloyd(gpu=True).fit(blobs(), 8, Budget(iterations=1), seed=0)


@pytest.mark.parametrize("fn,needle", [
    (faiss_official.omeans_available, "o-means authors' fork"),
    (flash_kmeans.is_available, "flash_kmeans"),
])
def test_unavailable_adapters_say_why(fn, needle):
    ok, why = fn()
    assert not ok and needle in why


def test_unavailable_adapters_raise_cleanly_and_are_flagged_unverified():
    X = blobs()
    for key in ("official_omeans", "flash_kmeans", "faiss_lloyd_gpu"):
        with pytest.raises(AdapterUnavailable):
            get_adapter(key).fit(X, 8, Budget(epochs=2), seed=0)
    st = {s.key: s for s in adapter_status()}
    assert not st["official_omeans"].verified and not st["flash_kmeans"].verified
    with pytest.raises(KeyError):
        get_adapter("nope")


# ------------------------------------------------------------------ isolation (A36)
def test_parent_process_never_imports_faiss():
    code = ("import sys, keo.algos.adapters.registry as r; r.adapter_status(); "
            "from keo.algos import Lloyd, EMeans, OMeans; "
            "import keo.runtime as rt; rt.hardware_descriptor(); print('faiss' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_worker_unknown_algo_and_failure_are_reported():
    with pytest.raises(RuntimeError, match="unknown algo"):
        wc.run_worker({"algo": "does_not_exist"})


# ------------------------------------------------------------------ official recipe == shipped code
GRID = [(1_000_000, 16_384, 3.0, 100), (1_183_514, 4_096, 2.0, 100), (1_183_514, 32_768, 2.0, 100),
        (50_000, 512, 3.0, 100), (200_000, 64, 1.5, 8), (10_000_000, 65_536, 2.5, 768), (30_000, 20, 5.0, 2)]


def test_official_recipe_matches_shipped_code():
    """Our `laws.official_recipe` must reproduce the SHIPPED faiss.contrib.e_means recipe exactly (B and alpha0)."""
    res, _, _ = wc.run_worker({"algo": "recipe", "cases": [list(c) for c in GRID]})
    for case, shipped in zip(GRID, res["recipes"]):
        B, a0, law = laws.official_recipe(*case[:3], case[3])
        assert (B, law) == (shipped["batch_size"], shipped["law"]), (case, shipped)
        assert a0 == pytest.approx(shipped["alpha0"], rel=1e-12), (case, shipped)


# ------------------------------------------------------------------ official e-means (real run)
def test_official_emeans_runs_with_shared_init_and_reports_recipe():
    X = blobs()
    k = 16
    C0, _ = random_row_init(X, k, 0)
    r = faiss_official.OfficialEmeans().fit(X, k, Budget(epochs=3), seed=0, init_centroids=C0)
    d = r.diagnostics
    assert r.centroids.shape == (k, X.shape[1]) and np.isfinite(r.centroids).all()
    assert d["batch_size"] and 0 < d["alpha0"] <= 1 and d["variant"].endswith("NumpyEmeans")
    assert r.train_labels is not None and r.train_labels.shape == (X.shape[0],)
    n = X.shape[0]
    assert r.op_counts["assign_passes"] == pytest.approx(3.0, abs=0.01)  # 3 epochs of rows
    assert r.op_counts["distance_computations"] == pytest.approx(3.0 * n * k, rel=0.01)
    again = faiss_official.OfficialEmeans().fit(X, k, Budget(epochs=3), seed=0, init_centroids=C0)
    np.testing.assert_array_equal(r.centroids, again.centroids)  # deterministic given seed + init


def test_official_emeans_requires_epochs_budget():
    with pytest.raises(ValueError):
        faiss_official.OfficialEmeans().fit(blobs(), 8, Budget(iterations=3), seed=0)


# ------------------------------------------------------------------ recipe switch on the reference backend
def test_reference_official_recipe_uses_shipped_constants():
    X = blobs()
    n, k, d = X.shape[0], 16, X.shape[1]
    r = EMeans(recipe="official").fit(X, k, Budget(epochs=3), seed=0)
    B, a0, law = laws.official_recipe(n, k, 3.0, d)
    dg = r.diagnostics
    assert (dg["recipe"], dg["batch_size"], dg["batch_law"]) == ("official", B, law)
    assert dg["alpha0"] == pytest.approx(a0) and dg["n_init"] == 2.0**-16 and dg["split_delta"] == 1e-4
    assert EMeans().fit(X, k, Budget(epochs=3), seed=0).diagnostics["n_init"] == 2.0**-16  # default = official (A52)
    assert EMeans(recipe="paper").fit(X, k, Budget(epochs=3), seed=0).diagnostics["n_init"] == 1e-10
    with pytest.raises(ValueError):
        EMeans(recipe="x")


def test_official_recipe_switches_revival_off_for_d_le_2():
    X = np.random.default_rng(0).standard_normal((4000, 2)).astype(np.float32)
    r = EMeans(recipe="official").fit(X, 32, Budget(epochs=2), seed=0)
    assert r.diagnostics["revival"] == "off" and r.diagnostics["n_revived_total"] == 0
