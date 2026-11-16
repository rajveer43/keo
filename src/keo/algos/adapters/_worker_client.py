"""Parent-side client for `_faiss_worker.py` (runs it in a separate process; the parent never imports faiss)."""

from __future__ import annotations

import functools
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

WORKER = Path(__file__).with_name("_faiss_worker.py")
UNAVAILABLE_EXIT = 3


class AdapterUnavailable(RuntimeError):
    """The requested official implementation cannot run here (missing package, fork, or GPU)."""


def faiss_installed() -> bool:
    return importlib.util.find_spec("faiss") is not None


def run_worker(
    params: dict[str, Any],
    X: np.ndarray | None = None,
    init: np.ndarray | None = None,
    want_labels: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    """Run one worker job. Returns (result dict, centroids or None, labels or None)."""
    if not faiss_installed():
        raise AdapterUnavailable("faiss is not installed (pip install faiss-cpu)")
    with tempfile.TemporaryDirectory(prefix="keo_worker_") as tmp:
        w = Path(tmp)
        if X is not None:
            np.save(w / "X.npy", np.ascontiguousarray(X, dtype=np.float32))
        if init is not None:
            np.save(w / "init.npy", np.ascontiguousarray(init, dtype=np.float32))
        (w / "params.json").write_text(json.dumps(params))
        proc = subprocess.run([sys.executable, "-I", str(WORKER), str(w)], capture_output=True, text=True)
        res: dict[str, Any] = {}
        if (w / "result.json").exists():
            res = json.loads((w / "result.json").read_text())
        if proc.returncode == UNAVAILABLE_EXIT:
            raise AdapterUnavailable(res.get("unavailable", "unavailable"))
        if proc.returncode != 0:
            raise RuntimeError(
                f"worker failed (algo={params.get('algo')}, exit {proc.returncode}): "
                f"{res.get('error') or proc.stderr[-2000:]}"
            )
        cen = np.load(w / "centroids.npy") if (w / "centroids.npy").exists() else None
        lab = np.load(w / "labels.npy") if (want_labels and (w / "labels.npy").exists()) else None
    return res, cen, lab


@functools.lru_cache(maxsize=1)
def probe() -> dict[str, Any]:
    """Capabilities of the installed faiss (cached per process): version, e_means/o_means presence, GPU count."""
    res, _, _ = run_worker({"algo": "probe"})
    return res
