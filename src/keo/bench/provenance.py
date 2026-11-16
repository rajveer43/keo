"""Run provenance: git sha (if any), peak memory, timestamps."""

from __future__ import annotations

import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from keo.paths import project_root


def git_sha(root: Path | None = None) -> str | None:
    """HEAD sha if the project is a git checkout; None otherwise (this project is not a git repo by default)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root or project_root()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def peak_rss_mb() -> float:
    """Process high-water-mark resident memory so far (monotone; NOT per-run). macOS reports bytes, Linux KiB (A53)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(r) / 1024.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
