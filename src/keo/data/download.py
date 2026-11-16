"""Dataset download with size + SHA-256 verification.

Downloaded files are untrusted data: each dataset gets its own directory under data/<name>/ and is
only ever parsed by h5py / numpy (never executed). ann-benchmarks publishes no checksums, so the
first download is trust-on-first-use (TOFU): the digest is recorded in data/<name>/MANIFEST.json and
should then be pinned in the dataset YAML (`source.sha256`). Once pinned, a mismatch is a hard error.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from keo.config import DatasetConfig
from keo.paths import data_dir

CHUNK = 1 << 20  # 1 MiB


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    nbytes: int
    pinned: bool  # True if the digest was checked against a pinned value
    downloaded: bool  # False if the file was already present and verified


def dataset_dir(cfg: DatasetConfig) -> Path:
    return data_dir() / cfg.name


def dataset_file(cfg: DatasetConfig) -> Path:
    return dataset_dir(cfg) / cfg.source.filename


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_file(cfg: DatasetConfig, path: Path | None = None) -> tuple[bool, str]:
    """Check an existing file against expected_bytes / pinned sha256. Returns (ok, message)."""
    path = path or dataset_file(cfg)
    if not path.is_file():
        return False, f"missing: {path}"
    size = path.stat().st_size
    if cfg.source.expected_bytes is not None and size != cfg.source.expected_bytes:
        return False, f"size mismatch: {size} != expected {cfg.source.expected_bytes}"
    digest = sha256_file(path)
    if cfg.source.sha256:
        if digest != cfg.source.sha256:
            return False, f"sha256 mismatch: {digest} != pinned {cfg.source.sha256}"
        return True, f"ok (sha256 pinned, {size} bytes)"
    return True, f"ok but UNPINNED (sha256={digest}, {size} bytes)"


def _write_manifest(cfg: DatasetConfig, sha: str, nbytes: int) -> None:
    manifest = {
        "name": cfg.name,
        "url": cfg.source.url,
        "filename": cfg.source.filename,
        "sha256": sha,
        "bytes": nbytes,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "trust-on-first-use digest; pin in configs/datasets/<name>.yaml",
    }
    (dataset_dir(cfg) / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def download(cfg: DatasetConfig, force: bool = False, quiet: bool = False) -> DownloadResult:
    out_dir = dataset_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = dataset_file(cfg)

    if final.is_file() and not force:
        ok, msg = verify_file(cfg, final)
        if ok:
            digest = sha256_file(final)
            return DownloadResult(final, digest, final.stat().st_size, bool(cfg.source.sha256), False)
        raise DownloadError(f"existing file failed verification ({msg}); re-run with --force to replace it")

    part = final.with_suffix(final.suffix + ".part")
    h = hashlib.sha256()
    written = 0
    t0 = time.time()
    req = urllib.request.Request(cfg.source.url, headers={"User-Agent": "keo-benchmark/0.0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                h.update(chunk)
                written += len(chunk)
                if not quiet and sys.stderr.isatty():
                    pct = f"{100 * written / total:5.1f}%" if total else "  ?  "
                    rate = written / max(time.time() - t0, 1e-6) / 1e6
                    print(f"\r  {written / 1e6:8.1f} MB {pct} {rate:6.1f} MB/s", end="", file=sys.stderr)
            if total and written != total:
                raise DownloadError(f"truncated download: got {written} of {total} bytes")
    except Exception:
        part.unlink(missing_ok=True)
        raise
    finally:
        if not quiet and sys.stderr.isatty():
            print(file=sys.stderr)

    digest = h.hexdigest()
    if cfg.source.expected_bytes is not None and written != cfg.source.expected_bytes:
        part.unlink(missing_ok=True)
        raise DownloadError(f"size mismatch: {written} != expected {cfg.source.expected_bytes}")
    if cfg.source.sha256 and digest != cfg.source.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(f"sha256 mismatch: got {digest}, pinned {cfg.source.sha256}")

    part.replace(final)
    _write_manifest(cfg, digest, written)
    return DownloadResult(final, digest, written, bool(cfg.source.sha256), True)
