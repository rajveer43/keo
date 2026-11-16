"""Benchmark engine: schema, store, experiment expansion, runner (resume, failures, baselines), CLI. Synthetic data only."""

import json

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from keo.algos import Budget, Lloyd
from keo.algos.init import random_row_init
from keo.bench.experiment import (
    BASELINE_REF_KEY, expand, load_experiment_config, make_cell_id, parse_experiment_config,
)
from keo.bench.runner import make_algo, run_experiment, summarize
from keo.bench.schema import COLUMNS, SCHEMA, new_row, validate_row
from keo.bench.store import ResultStore
from keo.cli import app
from keo.config import ConfigError, parse_dataset_config
from keo.data.prep import prepare, subset_train
from keo.metrics import mse

N_TRAIN, N_EVAL, D = 4000, 400, 8


@pytest.fixture()
def ds(tmp_path, monkeypatch):
    """A tiny hdf5 dataset (8 Gaussian blobs, d=8) registered under KEO_DATA_DIR."""
    monkeypatch.setenv("KEO_DATA_DIR", str(tmp_path / "data"))
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, D)).astype(np.float32) * 6

    def draw(n):
        return (centers[rng.integers(0, 8, n)] + rng.standard_normal((n, D))).astype(np.float32)

    d = tmp_path / "data" / "toy"
    d.mkdir(parents=True)
    with h5py.File(d / "toy.hdf5", "w") as f:
        f["train"], f["test"] = draw(N_TRAIN), draw(N_EVAL)
    return parse_dataset_config({
        "name": "toy", "d": D, "n_train": N_TRAIN, "n_eval": N_EVAL, "normalize": False,
        "source": {"type": "url", "url": "file:///unused", "filename": "toy.hdf5", "format": "hdf5",
                   "train_key": "train", "eval_key": "test"},
    })


def exp_cfg(**over):
    raw = {
        "name": "t", "dataset": "toy", "n": [2000], "k": [16], "seeds": [0, 1], "warmup": False,
        "runs": [
            {"algo": "lloyd", "backend": "reference", "budgets": [1, 3, 25]},
            {"algo": "emeans", "backend": "reference", "budgets": [1, 2]},
            {"algo": "omeans", "backend": "reference", "budgets": [2], "params": {"cell_size": 4}},
        ],
    }
    raw.update(over)
    return parse_experiment_config(raw)


def QUIET(*_):
    return None


# ------------------------------------------------------------------ schema
def test_schema_columns_unique_and_blank_row_valid():
    assert len(COLUMNS) == len(set(COLUMNS)) == len(SCHEMA.names)
    validate_row(new_row())  # everything nullable


def test_schema_rejects_unknown_missing_and_wrong_types():
    r = new_row()
    r["bogus"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_row(r)
    r = new_row()
    del r["test_mse"]
    with pytest.raises(ValueError, match="missing"):
        validate_row(r)
    r = new_row()
    r["n"] = "not a number"
    with pytest.raises(ValueError, match="schema"):
        validate_row(r)


# ------------------------------------------------------------------ experiment config + expansion
def test_experiment_config_validation():
    base = {"name": "x", "dataset": "d", "n": [100], "k": [4], "seeds": [0], "runs": [{"algo": "lloyd", "budgets": [1]}]}
    parse_experiment_config(base)
    bad = [
        {"runs": [{"algo": "nope", "budgets": [1]}]},
        {"runs": [{"algo": "lloyd", "budgets": []}]},
        {"runs": [{"algo": "lloyd", "budgets": [1.5]}]},  # iterations must be integers
        {"runs": [{"algo": "emeans", "budgets": [0]}]},
        {"runs": [{"algo": "emeans", "backend": "reference", "adapter": "x", "budgets": [1]}]},
        {"runs": [{"algo": "emeans", "backend": "mars", "budgets": [1]}]},
        {"seeds": []},
    ]
    for over in bad:
        with pytest.raises(ConfigError):
            parse_experiment_config({**base, **over})
    with pytest.raises(ConfigError, match="missing required key 'runs'"):
        parse_experiment_config({k: v for k, v in base.items() if k != "runs"})


def test_smoke_config_loads():
    e = load_experiment_config("smoke")
    assert e.name == "smoke" and e.k_grid == (512,) and len(e.runs) == 6


def test_expand_merges_duplicate_specs_and_adds_baseline(ds):
    e = exp_cfg(runs=[{"algo": "emeans", "budgets": [1, 2]}, {"algo": "emeans", "budgets": [2, 3]}])
    plan = expand(e, ds)
    em = [u for u in plan.units if u.spec.algo == "emeans"]
    assert len(em) == 2 and all(u.budgets == [1.0, 2.0, 3.0] for u in em)  # merged, one unit per seed
    base = [u for u in plan.units if u.spec.key == BASELINE_REF_KEY]
    assert len(base) == 2 and all(u.baseline and u.budgets == [25] for u in base)


def test_expand_sorts_baseline_first_within_each_cell_group(ds):
    e = exp_cfg(runs=[{"algo": "lloyd", "budgets": [3], "params": {"init": "kmeanspp"}},
                      {"algo": "lloyd", "budgets": [25]}, {"algo": "emeans", "budgets": [1]}])
    plan = expand(e, ds)
    firsts = {}
    for u in plan.units:
        firsts.setdefault((u.n, u.k, u.seed), u)
    assert all(u.spec.key == BASELINE_REF_KEY for u in firsts.values())


def test_expand_skips_invalid_nk_and_unavailable_adapters_with_reasons(ds):
    e = exp_cfg(k=[16, 1000], n=[2000, None],
                runs=[{"algo": "lloyd", "budgets": [25]}, {"algo": "omeans", "backend": "official", "budgets": [1]}])
    plan = expand(e, ds, availability={"official_omeans": (False, "no fork")})
    reasons = [s["reason"] for s in plan.skipped]
    assert any("n/k <" in r for r in reasons) and any("no fork" in r for r in reasons)
    # k=1000 violates n/k >= 8 at both n=2000 and n=4000; only k=16 survives
    assert {(u.n, u.k) for u in plan.units} == {(2000, 16), (N_TRAIN, 16)}


def test_cell_ids_are_stable_and_distinguish_every_axis():
    key = BASELINE_REF_KEY
    base = make_cell_id("d", False, 100, 4, key, 25, 0)
    assert base == make_cell_id("d", False, 100, 4, key, 25, 0)
    others = {make_cell_id("d2", False, 100, 4, key, 25, 0), make_cell_id("d", True, 100, 4, key, 25, 0),
              make_cell_id("d", False, 101, 4, key, 25, 0), make_cell_id("d", False, 100, 5, key, 25, 0),
              make_cell_id("d", False, 100, 4, key, 10, 0), make_cell_id("d", False, 100, 4, key, 25, 1),
              make_cell_id("d", False, 100, 4, ("lloyd", "reference", None, '{"init":"kmeanspp"}'), 25, 0)}
    assert base not in others and len(others) == 7


# ------------------------------------------------------------------ store
def _row(cell, status="ok", **kw):
    r = new_row()
    r.update(cell_id=cell, status=status, experiment="e", test_mse=1.5, **kw)
    return r


def test_store_roundtrip_atomic_and_failed_then_ok(tmp_path):
    s = ResultStore(tmp_path)
    assert not s.done("c1")
    s.write(_row("c1", "failed", error="boom"))
    assert s.has_failed("c1") and s.done("c1") and not s.done("c1", retry_failed=True)
    s.write(_row("c1", "ok"))
    assert s.has_ok("c1") and not s.has_failed("c1")  # the ok row replaces the failed one
    assert s.read_row("c1")["test_mse"] == 1.5
    assert not list(s.rows_dir.glob("*.tmp"))
    s.write(_row("c2"))
    assert s.counts() == {"ok": 2, "failed": 0} and len(s.read_all()) == 2
    assert s.consolidate().exists()
    with pytest.raises(ValueError):
        s.write(_row("c3", status="weird"))


# ------------------------------------------------------------------ runner end to end
def test_end_to_end_rows_are_complete_valid_and_baseline_is_zero(ds, tmp_path):
    e = exp_cfg()
    out = tmp_path / "out"
    s = run_experiment(e, ds, out, log=QUIET)
    assert s.n_failed == 0 and s.n_ok == s.n_cells_planned == 2 * (3 + 2 + 1)
    df = ResultStore(out).read_all()
    assert set(df.status) == {"ok"} and df.cell_id.is_unique
    for col in ("train_mse", "val_mse", "test_mse", "t_fit", "distance_computations", "hardware_json", "rel_test_vs_lloyd25_ref"):
        assert df[col].notna().all(), col
    l25 = df[(df.algo == "lloyd") & (df.budget == 25)]
    assert (l25.rel_test_vs_lloyd25_ref == 0).all() and (l25.rel_train_vs_lloyd25_ref == 0).all()
    assert (l25.role == "baseline").all()
    # the o-means / e-means rows are relative to the Lloyd@25 of the same seed
    for _, r in df[df.algo != "lloyd"].iterrows():
        ref = l25[l25.seed == r.seed].test_mse.iloc[0]
        assert r.rel_test_vs_lloyd25_ref == pytest.approx((r.test_mse - ref) / ref)
    assert (df.backend == "reference").all() and set(df.variant) == {"reference"}
    assert df[df.algo == "omeans"].nlist.notna().all() and df[df.algo == "emeans"].nlist.isna().all()
    assert df[df.algo == "omeans"].assign_fraction_actual.notna().all()


def test_lloyd_snapshot_rows_equal_independent_runs(ds, tmp_path):
    e = exp_cfg(runs=[{"algo": "lloyd", "budgets": [1, 3, 25]}], seeds=[2])
    out = tmp_path / "out"
    run_experiment(e, ds, out, log=QUIET)
    df = ResultStore(out).read_all()
    prep = prepare(ds)
    X = subset_train(prep.train, 2000, seed=0)
    C0, _ = random_row_init(X, 16, 2)
    for it in (1, 3, 25):
        ind = Lloyd().fit(X, 16, Budget(iterations=it), seed=2, init_centroids=C0)
        row = df[(df.algo == "lloyd") & (df.budget == it)].iloc[0]
        assert row.test_mse == pytest.approx(mse(prep.test, ind.centroids), rel=1e-12)


def test_resume_is_a_noop_and_touches_nothing(ds, tmp_path):
    e, out = exp_cfg(), tmp_path / "out"
    first = run_experiment(e, ds, out, log=QUIET)
    files = {p.name: p.stat().st_mtime_ns for p in (out / "rows").glob("*.parquet")}
    second = run_experiment(e, ds, out, log=QUIET)
    assert second.n_ok == 0 and second.n_failed == 0 and second.n_cells_resumed == first.n_cells_planned
    assert files == {p.name: p.stat().st_mtime_ns for p in (out / "rows").glob("*.parquet")}


def test_resume_after_partial_run_only_runs_the_missing_cells(ds, tmp_path):
    e, out = exp_cfg(), tmp_path / "out"
    part = run_experiment(e, ds, out, limit=3, log=QUIET)
    assert 0 < part.n_ok < part.n_cells_planned
    rest = run_experiment(e, ds, out, log=QUIET)
    assert rest.n_cells_resumed == part.n_ok and part.n_ok + rest.n_ok == part.n_cells_planned


def test_deterministic_across_independent_runs(ds, tmp_path):
    e = exp_cfg()
    a, b = tmp_path / "a", tmp_path / "b"
    run_experiment(e, ds, a, log=QUIET)
    run_experiment(e, ds, b, log=QUIET)
    da, db = ResultStore(a).read_all().set_index("cell_id"), ResultStore(b).read_all().set_index("cell_id")
    for col in ("train_mse", "val_mse", "test_mse", "distance_computations"):
        np.testing.assert_array_equal(da[col].sort_index().to_numpy(), db[col].sort_index().to_numpy())


def test_failures_are_recorded_not_silent_and_retry_works(ds, tmp_path):
    bad = exp_cfg(runs=[{"algo": "omeans", "budgets": [1], "params": {"nlist": 64}}], seeds=[0])  # nlist > k=16
    out = tmp_path / "out"
    s = run_experiment(bad, ds, out, log=QUIET)
    assert s.n_failed == 1 and s.n_ok == 1  # failed o-means cell + the auto-added Lloyd@25 baseline
    df = ResultStore(out).read_all()
    f = df[df.status == "failed"].iloc[0]
    assert "nlist" in f.error and np.isnan(f.test_mse)
    again = run_experiment(bad, ds, out, log=QUIET)
    assert again.n_failed == 0 and again.n_cells_resumed == 2  # failures are not retried by default
    retry = run_experiment(bad, ds, out, retry_failed=True, log=QUIET)
    assert retry.n_failed == 1  # retried, still failing, still recorded
    assert ResultStore(out).counts() == {"ok": 1, "failed": 1}


def test_invalid_reference_params_fail_fast_as_config_error(ds, tmp_path):
    bad = exp_cfg(runs=[{"algo": "emeans", "budgets": [1], "params": {"not_a_param": 1}}])
    with pytest.raises(ConfigError, match="invalid params"):
        run_experiment(bad, ds, tmp_path / "o", log=QUIET)
    rows = tmp_path / "o" / "rows"
    assert not rows.exists() or not list(rows.glob("*.parquet"))


def test_dry_run_writes_no_rows(ds, tmp_path):
    s = run_experiment(exp_cfg(), ds, tmp_path / "o", dry_run=True, log=QUIET)
    assert s.dry_run and s.n_cells_planned == 12 and s.n_ok == 0
    assert not list((tmp_path / "o" / "rows").glob("*.parquet"))
    assert json.loads((tmp_path / "o" / "skipped.json").read_text()) == []


def test_official_lloyd_baseline_fills_the_official_relative_columns(ds, tmp_path):
    e = exp_cfg(runs=[{"algo": "lloyd", "backend": "official", "budgets": [25]}, {"algo": "emeans", "budgets": [2]}], seeds=[0])
    out = tmp_path / "o"
    s = run_experiment(e, ds, out, log=QUIET)
    assert s.n_failed == 0
    df = ResultStore(out).read_all()
    off = df[(df.algo == "lloyd") & (df.backend == "official")].iloc[0]
    assert off.rel_test_vs_lloyd25_official == 0 and off.variant == "faiss-cpu"
    em = df[df.algo == "emeans"].iloc[0]
    assert em.rel_test_vs_lloyd25_official == pytest.approx((em.test_mse - off.test_mse) / off.test_mse)
    assert abs(off.rel_test_vs_lloyd25_ref) < 5e-3  # Faiss Lloyd ~ reference Lloyd (Phase 2 parity)


def test_official_emeans_rows_are_flagged_official_and_adapter_recorded(ds, tmp_path):
    e = exp_cfg(runs=[{"algo": "emeans", "backend": "official", "budgets": [2]}], seeds=[0])
    out = tmp_path / "o"
    run_experiment(e, ds, out, log=QUIET)
    df = ResultStore(out).read_all()
    r = df[df.algo == "emeans"].iloc[0]
    assert r.backend == "official" and r.adapter == "official_emeans_cpu" and "NumpyEmeans" in r.variant


def test_unavailable_adapter_group_is_skipped_and_listed_not_failed(ds, tmp_path):
    e = exp_cfg(runs=[{"algo": "omeans", "backend": "official", "budgets": [1]}], seeds=[0])
    out = tmp_path / "o"
    s = run_experiment(e, ds, out, availability={"official_omeans": (False, "needs the fork + CUDA")}, log=QUIET)
    assert s.n_failed == 0 and any("needs the fork" in x["reason"] for x in s.skipped)
    assert "needs the fork" in (out / "skipped.json").read_text()


def test_make_algo_reference_and_official_selection():
    from keo.algos import EMeans
    from keo.bench.experiment import RunSpec

    assert isinstance(make_algo(RunSpec("emeans", "reference", (1.0,), {"recipe": "paper"})), EMeans)
    assert make_algo(RunSpec("lloyd", "official", (25,), {}, "faiss_lloyd_cpu")).backend == "official"


def test_summarize_groups_by_algo_backend_budget(ds, tmp_path):
    out = tmp_path / "o"
    run_experiment(exp_cfg(), ds, out, log=QUIET)
    tab = summarize(ResultStore(out).read_all())
    assert {"algo", "backend", "budget", "rel_test_vs_L25", "passes", "seeds"} <= set(tab.columns)
    assert (tab.seeds == 2).all()


# ------------------------------------------------------------------ CLI
def test_cli_run_dry_run_and_status(tmp_path, monkeypatch):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "name: cli\ndataset: glove100\nn: [50000]\nk: [512]\nseeds: [0]\nruns:\n  - {algo: lloyd, budgets: [1, 25]}\n")
    r = CliRunner().invoke(app, ["run", str(cfg), "--dry-run", "--out", str(tmp_path / "res")])
    assert r.exit_code == 0 and "DRY RUN cells planned=2" in r.output
    monkeypatch.setenv("KEO_RESULTS_DIR", str(tmp_path / "rr"))
    (tmp_path / "rr" / "x").mkdir(parents=True)
    assert "ok=0 failed=0" in CliRunner().invoke(app, ["status", "x"]).output


def test_cli_run_unknown_config_exit_2():
    assert CliRunner().invoke(app, ["run", "does_not_exist"]).exit_code == 2
