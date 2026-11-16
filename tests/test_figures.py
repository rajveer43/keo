"""Figure generation: files, CSV table twins trace to stored rows, graceful skips, CLI errors. Synthetic data only."""

import h5py
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from keo.analysis import figures as F
from keo.analysis.figures import FIGURES, THEMES, make_figures
from keo.bench.experiment import parse_experiment_config
from keo.bench.runner import run_experiment
from keo.bench.store import ResultStore
from keo.cli import app
from keo.config import parse_dataset_config

D = 8


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """One small experiment with everything the figures need: K/E/O reference, official K/E, 2 grid cells, 2 seeds."""
    root = tmp_path_factory.mktemp("fig")
    mp = pytest.MonkeyPatch()
    mp.setenv("KEO_DATA_DIR", str(root / "data"))
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, D)).astype(np.float32) * 6
    draw = lambda n: (centers[rng.integers(0, 8, n)] + rng.standard_normal((n, D))).astype(np.float32)  # noqa: E731
    (root / "data" / "toy").mkdir(parents=True)
    with h5py.File(root / "data" / "toy" / "toy.hdf5", "w") as f:
        f["train"], f["test"] = draw(4000), draw(400)
    ds = parse_dataset_config({"name": "toy", "d": D, "n_train": 4000, "n_eval": 400, "normalize": False,
                               "source": {"type": "url", "url": "file:///x", "filename": "toy.hdf5", "format": "hdf5",
                                          "train_key": "train", "eval_key": "test"}})
    exp = parse_experiment_config({
        "name": "figtest", "dataset": "toy", "n": [2000, 3000], "k": [16], "seeds": [0, 1], "warmup": False,
        "runs": [
            {"algo": "lloyd", "budgets": [1, 3, 10, 25]},
            {"algo": "emeans", "budgets": [1, 2, 3]},
            {"algo": "omeans", "budgets": [1, 2, 3], "params": {"cell_size": 4}},
            {"algo": "lloyd", "backend": "official", "budgets": [25]},
            {"algo": "emeans", "backend": "official", "budgets": [2]},
        ]})
    out = root / "out"
    s = run_experiment(exp, ds, out, log=lambda *_: None)
    assert s.n_failed == 0
    yield out
    mp.undo()


def test_all_figures_render_in_both_themes_with_csv_twins(results, tmp_path):
    df = ResultStore(results).read_all()
    written = make_figures(df, tmp_path, "figtest", themes=("light", "dark"))
    assert set(written) == set(FIGURES)
    for name, files in written.items():
        names = {f.name for f in files}
        assert {f"{name}.png", f"{name}_dark.png", f"{name}.csv"} <= names, name
        for png in (f for f in files if f.suffix == ".png"):
            assert png.stat().st_size > 5000
            img = mpimg.imread(png)
            assert img.ndim == 3 and img.shape[0] > 300 and img.shape[1] > 300
        assert len(pd.read_csv(tmp_path / f"{name}.csv")) >= 1  # the table twin has data rows


def test_light_and_dark_use_their_own_surface(results, tmp_path):
    df = ResultStore(results).read_all()
    make_figures(df, tmp_path, "figtest", themes=("light", "dark"), only=("convergence",))

    def corner(path):  # top-left pixel is the figure surface
        return (mpimg.imread(path)[2, 2, :3] * 255).round().astype(int)

    assert list(corner(tmp_path / "convergence.png")) == [0xFC, 0xFC, 0xFB]  # light surface #fcfcfb
    assert list(corner(tmp_path / "convergence_dark.png")) == [0x1A, 0x1A, 0x19]  # dark surface #1a1a19


def test_csv_twin_matches_stored_rows_exactly(results, tmp_path):
    df = ResultStore(results).read_all()
    make_figures(df, tmp_path, "figtest", only=("convergence",))
    tw = pd.read_csv(tmp_path / "convergence.csv")
    ok = df[df.status == "ok"]
    for _, r in tw.iterrows():
        g = ok[(ok.n == r.n) & (ok.k == r.k) & (ok.algo == r.algo) & (ok.backend == r.backend) & (ok.budget == r.budget)]
        assert len(g) == 2 and r.seeds == 2
        assert r.y == pytest.approx(100 * g.rel_test_vs_lloyd25_ref.median())
        assert r.lo == pytest.approx(100 * g.rel_test_vs_lloyd25_ref.min())  # < 5 seeds -> band is min-max
        assert r.hi == pytest.approx(100 * g.rel_test_vs_lloyd25_ref.max())
    assert set(tw.n) == {2000, 3000}  # one panel per (n, k)


def test_baseline_is_exactly_zero_in_the_plotted_data(results, tmp_path):
    df = ResultStore(results).read_all()
    make_figures(df, tmp_path, "figtest", only=("convergence",))
    tw = pd.read_csv(tmp_path / "convergence.csv")
    base = tw[(tw.algo == "lloyd") & (tw.backend == "reference") & (tw.budget == 25)]
    assert (base.y == 0).all()


def test_wallclock_never_mixes_backends_on_one_axis(results, tmp_path):
    df = ResultStore(results).read_all()
    make_figures(df, tmp_path, "figtest", only=("wallclock",))
    tw = pd.read_csv(tmp_path / "wallclock.csv")
    assert set(tw.backend) == {"reference", "official"}
    h = lambda name: mpimg.imread(tmp_path / name).shape[0]  # noqa: E731
    make_figures(df, tmp_path, "figtest", only=("convergence",))
    assert h("wallclock.png") > h("convergence.png")  # one panel row per backend makes it taller


def test_parity_pairs_same_seed_and_budget(results, tmp_path):
    df = ResultStore(results).read_all()
    make_figures(df, tmp_path, "figtest", only=("parity",))
    tw = pd.read_csv(tmp_path / "parity.csv")
    assert set(tw.algo) == {"lloyd", "emeans"}
    np.testing.assert_allclose(tw.diff_test_pct, 100 * (tw.test_mse - tw.test_mse_off) / tw.test_mse_off, rtol=1e-9)
    # NOTE: no agreement threshold here. On this n=2-3k toy the reference and official e-means differ by up to ~2 % on test
    # error (Lloyd agrees to ~1e-5). The 0.5 % agreement was shown at GloVe scale (tests/test_gate_phase5.py); this
    # test only checks that the figure pairs the right rows and computes the difference correctly.
    assert np.isfinite(tw.diff_test_pct).all() and len(tw) == 8  # (2 grid cells x 2 seeds) x {K@25, E@2}
    assert (tw[tw.algo == 'lloyd'].diff_test_pct.abs() < 1e-2).all()  # same init + deterministic Lloyd -> same answer


def test_figures_are_skipped_when_their_data_is_absent(results, tmp_path):
    df = ResultStore(results).read_all()
    no_official = df[df.backend == "reference"]
    w = make_figures(no_official, tmp_path / "a", "x")
    assert "parity" not in w and "omeans_cost" in w
    no_o = df[df.algo != "omeans"]
    w = make_figures(no_o, tmp_path / "b", "x")
    assert "omeans_cost" not in w and "convergence" in w
    failed_only = df.assign(status="failed")
    assert make_figures(failed_only, tmp_path / "c", "x") == {}


def test_figure_text_describes_but_never_asserts_a_finding():
    import inspect

    from keo.analysis import figures as F

    src = inspect.getsource(F)
    for banned in ("cross zero", "outperform", "beats k-means", "faster than"):
        assert banned not in src, f"a chart title/subtitle must not hard-code the finding '{banned}'"


def test_palette_slots_are_the_validated_ones():
    # validated with validate_palette.js --pairs all (slots 1-3), light + dark (docs/FIGURES.md)
    assert THEMES["light"].series == {"lloyd": "#2a78d6", "emeans": "#eb6834", "omeans": "#1baf7a"}
    assert THEMES["dark"].series == {"lloyd": "#3987e5", "emeans": "#d95926", "omeans": "#199e70"}
    assert THEMES["light"].surface == "#fcfcfb" and THEMES["dark"].surface == "#1a1a19"


# ------------------------------------------------------------------ CLI
def test_cli_figures_writes_files_and_reports_skips(results, tmp_path):
    r = CliRunner().invoke(app, ["figures", str(results), "--out", str(tmp_path / "f"), "--only", "convergence,parity"])
    assert r.exit_code == 0, r.output
    assert "convergence" in r.output and (tmp_path / "f" / "convergence.png").exists()
    assert not (tmp_path / "f" / "efficiency.png").exists()


def test_cli_figures_errors(tmp_path):
    assert CliRunner().invoke(app, ["figures", str(tmp_path / "missing")]).exit_code == 2
    assert CliRunner().invoke(app, ["figures", "smoke", "--theme", "neon"]).exit_code == 2


# ------------------------------------------------------------------ computed headlines (must follow the data, never assert a fixed finding)
def _synthetic(gaps, passes_scale=None, seconds=None, n=1000, k=10):
    """gaps: {algo: {budget: gap in percent}} -> a results frame with everything the headline helpers read."""
    rows = []
    for algo, per in gaps.items():
        for budget, gap in per.items():
            for seed in (0, 1):
                rows.append({"status": "ok", "n": n, "k": k, "algo": algo, "backend": "reference", "budget": budget,
                             "seed": seed, "rel_test_vs_lloyd25_ref": gap / 100.0, "rel_train_vs_lloyd25_ref": gap / 100.0,
                             "assign_passes": budget * (passes_scale or {}).get(algo, 1.0),
                             "t_fit": budget * (seconds or {}).get(algo, 0.01), "dataset": "glove100", "normalize": True,
                             "distance_computations": 100.0 * budget, "assign_fraction_actual": 0.5,
                             "assign_fraction_nominal": 0.4, "hardware_json": "{}"})
    return pd.DataFrame(rows)


GAPS = {"lloyd": {1: 5.0, 5: 0.8, 10: 0.3, 25: 0.0}, "emeans": {1: 1.0, 2: 0.1}, "omeans": {1: 0.9, 2: 0.1}}


def test_convergence_headline_is_computed_from_the_data():
    h = F.headline_convergence(_synthetic(GAPS))
    assert "e-means and o-means need 2 passes" in h and "k-means needs 10 passes" in h
    slower = {**GAPS, "emeans": {1: 2.0, 3: 0.2, 5: 0.1}}  # e-means now needs 3 passes: the sentence must change
    h2 = F.headline_convergence(_synthetic(slower))
    assert "o-means needs 2 passes" in h2 and "e-means needs 3 passes" in h2 and "k-means needs 10 passes" in h2


def test_headline_falls_back_to_a_plain_question_when_a_method_never_gets_close():
    never = {**GAPS, "omeans": {1: 3.0, 2: 2.0}}  # o-means never within 0.5 %
    assert F.headline_convergence(_synthetic(never)).startswith("How many passes")
    assert F.headline_efficiency(_synthetic(never)).startswith("How much work")
    assert F.headline_wallclock(_synthetic(never)).startswith("How long")
    assert F.headline_convergence(_synthetic({"lloyd": GAPS["lloyd"]})).startswith("How many passes")  # one method: no comparison


def test_efficiency_and_wallclock_headlines_use_their_own_quantities():
    df = _synthetic(GAPS, passes_scale={"omeans": 0.7}, seconds={"lloyd": 0.02, "emeans": 0.01, "omeans": 0.5})
    assert "o-means 1.4, e-means 2, k-means 10 full scans" in F.headline_efficiency(df)
    w = F.headline_wallclock(df)
    assert "e-means 20 ms" in w and "k-means 200 ms" in w and "o-means 1.0 s" in w  # budget x seconds-per-unit


def test_headline_threshold_is_the_documented_half_percent():
    assert F.GOOD == 0.5
    just_over = {**GAPS, "lloyd": {1: 5.0, 10: 0.51, 25: 0.0}}
    assert "k-means needs 25 passes" in F.headline_convergence(_synthetic(just_over))  # 0.51 % is not close enough


def test_omeans_cost_title_reports_the_measured_share():
    df = _synthetic(GAPS)
    df.loc[df.algo == "omeans", "distance_computations"] *= 0.7
    df.loc[df.algo == "emeans", "distance_computations"] *= 1.0
    df["distance_computations"] = df["distance_computations"]
    fig, table = F.fig_omeans_cost(df, THEMES["light"], "x")
    assert any("70%" in t.get_text() for t in fig.texts), [t.get_text() for t in fig.texts]
    assert len(table) >= 1


def test_parity_title_flips_when_the_gap_exceeds_the_agreement_limit():
    base = _synthetic({"emeans": {2: 0.0}, "lloyd": {25: 0.0}})
    off = base.assign(backend="official", test_mse=1.0, train_mse=1.0)
    ref_close = base.assign(test_mse=1.002, train_mse=1.0)
    fig, _ = F.fig_parity(pd.concat([ref_close, off], ignore_index=True), THEMES["light"], "x")
    assert any("matches the authors' code to within 0.2%" in t.get_text() for t in fig.texts)
    ref_far = base.assign(test_mse=1.02, train_mse=1.0)
    fig, _ = F.fig_parity(pd.concat([ref_far, off], ignore_index=True), THEMES["light"], "x")
    assert any("differs from the authors' code by up to 2%" in t.get_text() and "0.5%" in t.get_text() for t in fig.texts)


def test_no_internal_jargon_reaches_the_reader():
    import inspect

    src = inspect.getsource(F)
    for jargon in ("k-means@25", "L2-normalised", "min-max over seeds", "same seed", "Faiss, CPU", "torch", "(PyTorch"):
        assert jargon not in src, f"jargon '{jargon}' must not appear in chart text"
