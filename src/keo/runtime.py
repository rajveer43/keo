"""Thread control and environment descriptor (recorded in every result row from Phase 6 on)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def configure_threads(n: int | None) -> int:
    """Set the torch thread count. None -> leave default. Returns the effective count.

    Faiss threads are NOT set here: faiss runs in its own worker process (A36) and is configured through
    `FaissLloyd(threads=...)`. This module must never import faiss (libomp conflict with torch on macOS).
    """
    import torch

    if n is not None:
        torch.set_num_threads(int(n))
    return torch.get_num_threads()


def _cpu_brand() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def hardware_descriptor() -> dict[str, object]:
    import numpy
    import torch

    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_brand(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    import importlib.metadata as md

    for key, dist in (("faiss", "faiss-cpu"), ("sklearn", "scikit-learn")):
        try:
            info[key] = md.version(dist)  # metadata only: never import faiss here
        except md.PackageNotFoundError:
            info[key] = None
    return info
