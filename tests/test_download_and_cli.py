import dataclasses
import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from keo.cli import app
from keo.config import SourceConfig, load_dataset_config
from keo.data import download as dl

runner = CliRunner()


def _local_cfg(tmp_path: Path, payload: bytes, sha: str = "", nbytes=None):
    src = tmp_path / "remote.bin"
    src.write_bytes(payload)
    base = load_dataset_config("glove100")
    source = SourceConfig(
        type="url",
        url=src.as_uri(),
        filename="x.hdf5",
        format="hdf5",
        train_key="train",
        eval_key="test",
        sha256=sha,
        expected_bytes=nbytes,
    )
    return dataclasses.replace(base, name="toy_dl", source=source)


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KEO_DATA_DIR", str(tmp_path / "data"))


def test_download_unpinned_writes_manifest(tmp_path):
    payload = b"hello keo" * 1000
    cfg = _local_cfg(tmp_path, payload)
    res = dl.download(cfg, quiet=True)
    assert res.downloaded and not res.pinned
    assert res.sha256 == hashlib.sha256(payload).hexdigest()
    assert (dl.dataset_dir(cfg) / "MANIFEST.json").is_file()
    # second call: already present, verified
    again = dl.download(cfg, quiet=True)
    assert not again.downloaded and again.sha256 == res.sha256


def test_download_pinned_ok_and_mismatch(tmp_path):
    payload = b"abc" * 100
    good = hashlib.sha256(payload).hexdigest()
    res = dl.download(_local_cfg(tmp_path, payload, sha=good), quiet=True)
    assert res.pinned
    with pytest.raises(dl.DownloadError, match="sha256 mismatch"):
        dl.download(_local_cfg(tmp_path, payload, sha="0" * 64), force=True, quiet=True)
    # failed download leaves no partial file behind
    assert not list(dl.dataset_dir(_local_cfg(tmp_path, payload)).glob("*.part"))


def test_download_size_mismatch(tmp_path):
    with pytest.raises(dl.DownloadError, match="size mismatch"):
        dl.download(_local_cfg(tmp_path, b"x" * 10, nbytes=11), quiet=True)


def test_existing_corrupt_file_rejected(tmp_path):
    payload = b"abc" * 100
    cfg = _local_cfg(tmp_path, payload, sha=hashlib.sha256(payload).hexdigest())
    dl.download(cfg, quiet=True)
    dl.dataset_file(cfg).write_bytes(b"tampered")
    ok, msg = dl.verify_file(cfg)
    assert not ok and "mismatch" in msg
    with pytest.raises(dl.DownloadError, match="failed verification"):
        dl.download(cfg, quiet=True)


def test_cli_version_and_help():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and r.output.startswith("keo ")
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0 and "download" in r.output and "data-info" in r.output


def test_cli_list_and_unknown_dataset():
    r = runner.invoke(app, ["list-datasets"])
    assert "glove100" in r.output
    r = runner.invoke(app, ["download", "nope"])
    assert r.exit_code == 2


def test_cli_remaining_stubs_exit_3():
    # smoke / run were stubs until Phase 6; analyze / report arrive in Phase 8
    for cmd in (["analyze"], ["report"]):
        assert runner.invoke(app, cmd).exit_code == 3
