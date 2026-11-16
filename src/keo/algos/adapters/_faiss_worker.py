"""Stand-alone worker: runs Faiss-based algorithms in their OWN process.

Why a separate process: on macOS the torch and faiss wheels each bundle their own libomp; loading both into one
process aborts ("OMP: Error #15") or segfaults. KMP_DUPLICATE_LIB_OK=TRUE would silence that but can "silently
produce incorrect results", so we isolate instead (docs/ASSUMPTIONS.md A36).

Executed BY PATH (`python -I _faiss_worker.py <workdir>`), never imported, so it must not import torch (except in the
explicit GPU mode) or the `keo` package. Protocol: <workdir>/params.json {"algo": ..., ...}; optional X.npy, init.npy;
outputs result.json (+ centroids.npy, labels.npy). Exit code 3 = "algorithm unavailable" (reason in result.json).

Modes: kmeans | emeans | omeans | probe | recipe
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

UNAVAILABLE = 3


def _write(w: Path, obj: dict) -> None:
    (w / "result.json").write_text(json.dumps(obj, default=str))


def _load(w: Path):
    X = np.ascontiguousarray(np.load(w / "X.npy", mmap_mode="r"), dtype=np.float32)
    init = np.ascontiguousarray(np.load(w / "init.npy"), dtype=np.float32) if (w / "init.npy").exists() else None
    return X, init


def real_gpu_count(faiss) -> int:
    """GPUs usable by faiss. NOTE: the faiss-cpu wheel's `get_num_gpus()` returns 1 (a stub!), so a count is only
    trusted when the real GPU classes exist (found on faiss-cpu 1.16.0: `GpuIndexFlat` is absent)."""
    if not hasattr(faiss, "GpuIndexFlat") or not hasattr(faiss, "get_num_gpus"):
        return 0
    return int(faiss.get_num_gpus())


def run_probe(w: Path, p: dict) -> int:
    import faiss

    _write(w, {
        "faiss_version": faiss.__version__,
        "has_e_means": importlib.util.find_spec("faiss.contrib.e_means") is not None,
        "has_o_means": importlib.util.find_spec("faiss.contrib.o_means") is not None,
        "num_gpus": real_gpu_count(faiss),
        "raw_get_num_gpus": int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else None,
    })
    return 0


def run_recipe(w: Path, p: dict) -> int:
    """The shipped e-means recipe (batch size, alpha0, law) for given (n, k, T, d) -- used to verify our `official` recipe."""
    from faiss.contrib.e_means import base

    out = []
    for n, k, T, d in p["cases"]:
        law, _ = base._resolve_batch_law("auto", d)
        B, a0 = base._resolve_recipe(n, k, T, law, None, None)
        out.append({"n": n, "k": k, "T": T, "d": d, "law": law, "batch_size": int(B), "alpha0": float(a0)})
    _write(w, {"recipes": out})
    return 0


def run_kmeans(w: Path, p: dict) -> int:
    import faiss

    X, init = _load(w)
    n, d = X.shape
    if p.get("threads"):
        faiss.omp_set_num_threads(int(p["threads"]))
    use_gpu = bool(p.get("gpu"))
    if use_gpu and real_gpu_count(faiss) == 0:
        _write(w, {"unavailable": "faiss was built without GPU support or no GPU is visible"})
        return UNAVAILABLE
    km = faiss.Kmeans(
        d, int(p["k"]), niter=int(p["niter"]), nredo=1, seed=int(p["seed"]), verbose=False, gpu=use_gpu,
        max_points_per_centroid=2**30,  # disable the 256 points/centroid subsampling [E §4.1 fn.1]
        min_points_per_centroid=1,  # silence the warning-only <39 points/centroid notice
    )
    t0 = time.perf_counter()
    km.train(X, init_centroids=init)
    fit_s = time.perf_counter() - t0
    np.save(w / "centroids.npy", np.ascontiguousarray(km.centroids, dtype=np.float32))
    _write(w, {"fit_seconds": fit_s, "obj": [float(v) for v in km.obj], "faiss_version": faiss.__version__,
               "omp_threads": faiss.omp_get_max_threads(), "n": int(n)})
    return 0


def run_emeans(w: Path, p: dict) -> int:
    if importlib.util.find_spec("faiss.contrib.e_means") is None:
        _write(w, {"unavailable": "installed faiss has no contrib.e_means (needs faiss >= 1.16.0)"})
        return UNAVAILABLE
    import faiss
    from faiss.contrib.e_means import NumpyEmeans

    X, init = _load(w)
    n, d = X.shape
    if p.get("threads"):
        faiss.omp_set_num_threads(int(p["threads"]))
    extra = {kk: p[kk] for kk in ("batch_law", "unused_threshold", "batch_size", "alpha0") if p.get(kk) is not None}
    em = NumpyEmeans(d, int(p["k"]), nepochs=float(p["epochs"]), seed=int(p["seed"]), verbose=False, **extra)
    t0 = time.perf_counter()
    centroids, labels = em.train(X, init_centroids=init)
    fit_s = time.perf_counter() - t0
    np.save(w / "centroids.npy", np.ascontiguousarray(centroids, dtype=np.float32))
    if labels is not None:
        np.save(w / "labels.npy", np.asarray(labels, dtype=np.int64))
    rec = {kk: v for kk, v in (em.recipe or {}).items() if isinstance(v, (int, float, str, bool, list, type(None)))}
    _write(w, {"fit_seconds": fit_s, "recipe": rec, "n_splits": int(em.n_splits), "faiss_version": faiss.__version__,
               "omp_threads": faiss.omp_get_max_threads(), "n": int(n)})
    return 0


def run_omeans(w: Path, p: dict) -> int:
    """UNVERIFIED against a real install: written from the o-means project page (`Omeans(d, k, nepochs)`, `.train(x)`,
    `.assign(x)`); the authors' faiss fork has no public URL we could locate. CUDA only."""
    if importlib.util.find_spec("faiss.contrib.o_means") is None:
        _write(w, {"unavailable": "faiss.contrib.o_means not found: it lives in the o-means authors' faiss fork, not in "
                                  "the released faiss wheel"})
        return UNAVAILABLE
    import faiss  # noqa: F401  (import faiss before torch in this process)
    import torch

    if not torch.cuda.is_available():
        _write(w, {"unavailable": "o-means is a GPU algorithm and no CUDA device is visible"})
        return UNAVAILABLE
    from faiss.contrib.o_means import Omeans

    X, _ = _load(w)
    n, d = X.shape
    torch.manual_seed(int(p["seed"]))  # the page documents no seed/init argument: shared init is NOT possible
    x = torch.from_numpy(X).cuda()
    om = Omeans(d, int(p["k"]), nepochs=float(p["epochs"]))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    centroids, labels = om.train(x)
    torch.cuda.synchronize()
    fit_s = time.perf_counter() - t0
    np.save(w / "centroids.npy", np.ascontiguousarray(centroids.detach().cpu().numpy(), dtype=np.float32))
    _write(w, {"fit_seconds": fit_s, "n": int(n), "shared_init": False})
    return 0


MODES = {"probe": run_probe, "recipe": run_recipe, "kmeans": run_kmeans, "emeans": run_emeans, "omeans": run_omeans}


def main(workdir: str) -> int:
    w = Path(workdir)
    params = json.loads((w / "params.json").read_text())
    algo = params.get("algo", "kmeans")
    if algo not in MODES:
        _write(w, {"error": f"unknown algo {algo!r}"})
        return 2
    return MODES[algo](w, params)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
