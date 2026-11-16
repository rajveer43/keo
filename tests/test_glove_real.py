"""Gate test: the real GloVe-100 file loads and has the documented shape. Skipped if not downloaded."""

import pytest

from keo.config import load_dataset_config
from keo.data.download import dataset_file, verify_file
from keo.data.prep import prepare

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def cfg():
    c = load_dataset_config("glove100")
    if not dataset_file(c).is_file():
        pytest.skip("run `keo download glove100` first")
    return c


def test_file_verifies(cfg):
    ok, msg = verify_file(cfg)
    assert ok, msg


def test_prepare_shapes_and_splits(cfg):
    p = prepare(cfg)
    assert p.train.shape[1] == 100 and p.train.dtype.name == "float32"
    s = p.report["sizes"]
    assert s["val"] + s["test"] == p.report["dedup"]["eval_kept"] - p.report["zero_norm_dropped"]["eval"]
    assert abs(s["val"] - s["test"]) <= 1
    assert s["train"] > 1_000_000
