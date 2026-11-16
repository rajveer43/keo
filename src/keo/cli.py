"""`keo` command line. Phase 1 implements: download, verify, data-info, list-datasets.
Other commands are stubs that name the phase that delivers them."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from keo import __version__
from keo.config import ConfigError, list_dataset_configs, load_dataset_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="KEO: K-means vs E-means vs O-means benchmark.")


def _version_cb(value: bool) -> None:
    if value:
        typer.echo(f"keo {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[bool, typer.Option("--version", callback=_version_cb, is_eager=True, help="Show version.")] = False,
) -> None:
    pass


def _cfg(name: str):
    try:
        return load_dataset_config(name)
    except ConfigError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


@app.command("list-datasets")
def list_datasets() -> None:
    """List dataset configs found under configs/datasets/."""
    for n in list_dataset_configs():
        typer.echo(n)


@app.command()
def download(
    dataset: str,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if a verified file exists.")] = False,
) -> None:
    """Download a dataset into data/<name>/ and verify size/SHA-256."""
    from keo.data.download import DownloadError, download as do_download

    cfg = _cfg(dataset)
    try:
        res = do_download(cfg, force=force)
    except DownloadError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    state = "downloaded" if res.downloaded else "already present"
    typer.echo(f"{cfg.name}: {state}: {res.path}")
    typer.echo(f"  bytes  : {res.nbytes}")
    typer.echo(f"  sha256 : {res.sha256}")
    if res.pinned:
        typer.echo("  checksum verified against the value pinned in the dataset YAML")
    else:
        typer.secho(
            "  UNPINNED: digest recorded in MANIFEST.json (trust-on-first-use). "
            "Pin it by setting source.sha256 in the dataset YAML.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def verify(dataset: str) -> None:
    """Re-verify an already-downloaded dataset file."""
    from keo.data.download import verify_file

    cfg = _cfg(dataset)
    ok, msg = verify_file(cfg)
    typer.echo(msg)
    raise typer.Exit(0 if ok else 1)


@app.command("data-info")
def data_info(dataset: str, as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Load, dedup, (normalise), split; print shapes, norm statistics, dedup report and split sizes."""
    from keo.data.prep import prepare

    cfg = _cfg(dataset)
    prep = prepare(cfg)
    if as_json:
        typer.echo(json.dumps(prep.report, indent=2, default=str))
        return
    r = prep.report
    typer.echo(f"dataset          : {cfg.name}  (normalize={cfg.normalize})")
    typer.echo(f"raw train/eval   : {r['train_shape']} / {r['eval_shape']}")
    for split in ("train", "eval"):
        s = r[f"raw_norm_stats_{split}"]
        typer.echo(
            f"raw {split:5s} norms : min={s['min']:.4f} median={s['median']:.4f} mean={s['mean']:.4f} "
            f"max={s['max']:.4f} | within 1e-3 of 1: {100 * s['frac_within_1e-3_of_1']:.2f}% | zero: {s['n_zero']}"
        )
    typer.echo(f"train internal duplicate rows (info only, kept): {r['train_internal_duplicates']}")
    dd = r["dedup"]
    typer.echo(
        f"dedup (eval vs train): in={dd['eval_in']} removed_in_train={dd['eval_removed_present_in_train']} "
        f"removed_internal={dd['eval_removed_internal_duplicates']} kept={dd['eval_kept']}"
    )
    if cfg.normalize:
        typer.echo(f"zero-norm rows dropped: {r['zero_norm_dropped']}")
    s = r["sizes"]
    typer.echo(f"split sizes      : train={s['train']} val={s['val']} test={s['test']}")


def _run_and_report(config: str, out: str | None, retry_failed: bool, limit: int | None, dry_run: bool) -> int:
    from pathlib import Path

    from keo.bench.experiment import load_experiment_config
    from keo.bench.runner import run_experiment, summarize
    from keo.bench.store import ResultStore
    from keo.paths import results_dir

    try:
        exp = load_experiment_config(config)
        ds = load_dataset_config(exp.dataset)
    except ConfigError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        return 2
    out_dir = Path(out) if out else results_dir() / exp.name
    typer.echo(f"experiment '{exp.name}' on {ds.name} -> {out_dir}")
    try:
        summary = run_experiment(exp, ds, out_dir, retry_failed=retry_failed, limit=limit, dry_run=dry_run, log=typer.echo)
    except ConfigError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        return 2
    typer.echo(summary.line())
    if dry_run:
        for s in summary.skipped:
            typer.echo(f"  would skip: {s}")
        return 0
    df = ResultStore(out_dir).read_all()
    tab = summarize(df)
    if not tab.empty:
        typer.echo("\nmedian over seeds (rel. to the reference Lloyd@25 of the same seed; negative = better):")
        typer.echo(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if summary.n_failed:
        typer.secho(f"{summary.n_failed} cell(s) FAILED: see status/error columns in {out_dir}/results.parquet", fg=typer.colors.RED)
        return 1
    return 0


@app.command()
def run(
    config: str,
    out: Annotated[str | None, typer.Option(help="Output directory (default results/<experiment name>).")] = None,
    retry_failed: Annotated[bool, typer.Option("--retry-failed", help="Re-run cells whose stored row is a failure.")] = False,
    limit: Annotated[int | None, typer.Option(help="Stop after this many fit units (for trial runs).")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the plan (cells, skips) without running anything.")] = False,
) -> None:
    """Run an experiment grid. Resumable: stored cells are skipped."""
    raise typer.Exit(_run_and_report(config, out, retry_failed, limit, dry_run))


@app.command()
def smoke(
    out: Annotated[str | None, typer.Option(help="Output directory (default results/smoke).")] = None,
) -> None:
    """Tiny end-to-end run on a GloVe subset (CPU, < 5 min)."""
    raise typer.Exit(_run_and_report("smoke", out, False, None, False))


@app.command()
def status(experiment: str) -> None:
    """Counts of stored ok/failed cells for an experiment directory (results/<name>)."""
    from keo.bench.store import ResultStore
    from keo.paths import results_dir

    c = ResultStore(results_dir() / experiment).counts()
    typer.echo(f"{experiment}: ok={c['ok']} failed={c['failed']}")


@app.command()
def figures(
    experiment: str,
    theme: Annotated[str, typer.Option(help="light | dark | both")] = "light",
    out: Annotated[str | None, typer.Option(help="Output directory (default results/<experiment>/figures).")] = None,
    only: Annotated[str | None, typer.Option(help="Comma-separated figure names to render.")] = None,
) -> None:
    """Render the standard figure set (PNG + CSV table twin) from an experiment's stored results."""
    from pathlib import Path

    from keo.analysis.figures import FIGURES, make_figures
    from keo.bench.store import ResultStore
    from keo.paths import results_dir

    if theme not in {"light", "dark", "both"}:
        typer.secho("error: --theme must be light, dark or both", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    root = Path(experiment) if Path(experiment).is_dir() else results_dir() / experiment
    if not (root / "rows").is_dir():
        typer.secho(f"error: no results found at {root} (run `keo run`/`keo smoke` first)", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    df = ResultStore(root).read_all()
    if df.empty or not (df["status"] == "ok").any():
        typer.secho("error: no successful result rows to plot", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    names = tuple(x.strip() for x in only.split(",")) if only else None
    if names and (bad := [n for n in names if n not in FIGURES]):
        typer.secho(f"error: unknown figure(s) {bad}; known: {sorted(FIGURES)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    themes = ("light", "dark") if theme == "both" else (theme,)
    written = make_figures(df, Path(out) if out else root / "figures", root.name, themes, names)
    for name in FIGURES:
        if names and name not in names:
            continue
        if name in written:
            typer.echo(f"  {name:<14} " + ", ".join(p.name for p in written[name]))
        else:
            typer.echo(f"  {name:<14} (skipped: the data needed for this figure is not in this experiment)")


@app.command()
def adapters(as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Show which official implementations (Faiss, e-means, o-means, Flash-KMeans) can run on this machine."""
    from keo.algos.adapters.registry import adapter_status

    rows = adapter_status()
    if as_json:
        typer.echo(json.dumps([r.as_dict() for r in rows], indent=2))
        return
    for r in rows:
        mark = typer.style("available  ", fg=typer.colors.GREEN) if r.available else typer.style("unavailable", fg=typer.colors.YELLOW)
        ver = "verified API" if r.verified else "UNVERIFIED (written from docs)"
        typer.echo(f"{mark}  {r.key:<20} [{r.algo:<6}] {ver}")
        typer.echo(f"             {r.description}: {r.reason}")


def _stub(phase: int, what: str) -> None:
    typer.secho(f"not implemented yet: {what} arrives in Phase {phase}.", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(3)


@app.command()
def analyze() -> None:
    """Aggregate results (Phase 8)."""
    _stub(8, "`keo analyze`")


@app.command()
def report() -> None:
    """Generate REPORT.md (Phase 8)."""
    _stub(8, "`keo report`")


if __name__ == "__main__":
    app()
