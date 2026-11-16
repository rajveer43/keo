import pytest

from keo.config import ConfigError, list_dataset_configs, load_dataset_config, parse_dataset_config


def _raw(**over):
    raw = {
        "name": "toy",
        "source": {
            "type": "url",
            "url": "http://example.invalid/x.hdf5",
            "filename": "x.hdf5",
            "format": "hdf5",
            "train_key": "train",
            "eval_key": "test",
        },
        "d": 4,
        "n_train": 10,
        "n_eval": 4,
    }
    raw.update(over)
    return raw


def test_glove_config_loads():
    cfg = load_dataset_config("glove100")
    assert cfg.name == "glove100"
    assert (cfg.d, cfg.n_train, cfg.n_eval) == (100, 1_183_514, 10_000)
    assert cfg.normalize is True
    assert cfg.k_grid == (4096, 16384, 32768)
    assert cfg.n_subset_grid[-1] is None
    assert cfg.split.val_fraction == 0.5


def test_registry_lists_glove():
    assert "glove100" in list_dataset_configs()


def test_defaults_applied():
    cfg = parse_dataset_config(_raw())
    assert cfg.normalize is True and cfg.split.seed == 0 and cfg.min_points_per_cluster == 8


@pytest.mark.parametrize("missing", ["name", "source", "d", "n_train", "n_eval"])
def test_missing_required_key(missing):
    raw = _raw()
    del raw[missing]
    with pytest.raises(ConfigError, match=missing):
        parse_dataset_config(raw)


def test_bad_source_type_and_format():
    raw = _raw()
    raw["source"]["type"] = "ftp"
    with pytest.raises(ConfigError):
        parse_dataset_config(raw)
    raw = _raw()
    raw["source"]["format"] = "csv"
    with pytest.raises(ConfigError):
        parse_dataset_config(raw)


def test_bad_val_fraction():
    with pytest.raises(ConfigError):
        parse_dataset_config(_raw(split={"val_fraction": 1.5}))


def test_unknown_dataset_name():
    with pytest.raises(ConfigError, match="not found"):
        load_dataset_config("does_not_exist")
