"""Figures for KEO results, written for a first-time reader.

Reads the per-cell result rows of any experiment and writes PNG + a CSV "table twin" with exactly the plotted numbers.

Reading design (every figure):
  * a bold HEADLINE computed from the data (never hard-coded; if the data does not support a clear statement it falls back
    to a plain question), one short "how to read it" line, a shaded "close enough" zone so the goal is visible, and a one-line
    footer. Plain words: "groups" not k, "repeats" not seeds, "our code / authors' code" not reference / official.
  * the vertical axis is the GAP to the best k-means answer (k-means run for 25 rounds, same data, same starting point):
    0% = as good, higher = worse. Scale is stretched near 0% so small gaps stay visible (said in the footer).
Colour design (dataviz method; palette validated with `validate_palette.js --pairs all`, light + dark):
  * colour = the algorithm, marker shape = the algorithm too (circle / square / triangle), hollow = the authors' own code;
  * one axis per panel, hairline solid grid, 2 px lines, ringed >= 8 px markers, text in ink tokens only;
  * wall-clock is only ever plotted within ONE implementation (separate panel rows).
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ALGOS = ("lloyd", "emeans", "omeans")
NAME = {"lloyd": "k-means", "emeans": "e-means", "omeans": "o-means"}
LEGEND = {"lloyd": "k-means (the standard)", "emeans": "e-means", "omeans": "o-means"}
MARKER = {"lloyd": "o", "emeans": "s", "omeans": "^"}
LABEL_OFFSET = {"lloyd": (-10, 9, "right"), "emeans": (10, -16, "left"), "omeans": (-14, -22, "right")}
FRIENDLY_DATA = {"glove100": "GloVe word vectors"}
GOOD = 0.5  # percent: a method is "close enough" when within this gap of k-means' best answer
CLOSE = f"within {GOOD:g}% of k-means run for 25 rounds"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: dict[str, str]


THEMES = {
    "light": Theme("light", "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
                   {"lloyd": "#2a78d6", "emeans": "#eb6834", "omeans": "#1baf7a"}),
    "dark": Theme("dark", "#1a1a19", "#0d0d0d", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
                  {"lloyd": "#3987e5", "emeans": "#d95926", "omeans": "#199e70"}),
}


def _font() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "SF Pro Text", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if name in have:
            return name
    return "DejaVu Sans"


# ----------------------------------------------------------------------------------------------- data helpers
def ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"].copy()


def _combos(df: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted({(int(n), int(k)) for n, k in zip(df["n"], df["k"])})


def _agg(df: pd.DataFrame, x: str, y: str, scale: float = 1.0) -> pd.DataFrame:
    """Median over repeats of x and y per (algo, backend, budget), plus the spread (min-max if < 5 repeats, else middle half)."""
    rows = []
    for (algo, backend, budget), g in df.groupby(["algo", "backend", "budget"]):
        ns = g["seed"].nunique()
        yy = g[y].astype(float) * scale
        lo, hi = (yy.min(), yy.max()) if ns < 5 else (yy.quantile(0.25), yy.quantile(0.75))
        rows.append({"algo": algo, "backend": backend, "budget": budget, "x": float(g[x].astype(float).median()),
                     "y": float(yy.median()), "lo": float(lo), "hi": float(hi), "seeds": int(ns)})
    cols = ["algo", "backend", "budget", "x", "y", "lo", "hi", "seeds"]
    return pd.DataFrame(rows).sort_values(["algo", "backend", "budget"]) if rows else pd.DataFrame(columns=cols)


def _fmt_num(v: float) -> str:
    return f"{int(v):,}" if float(v).is_integer() else f"{v:g}"


def _passes(v: float) -> str:
    return f"{_fmt_num(v)} pass" + ("" if v == 1 else "es")


def _scans(v: float) -> str:
    return f"{v:.2g} full scans" if v != 1 else "1 full scan"


def _secs(v: float) -> str:
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.1f} s"


# ----------------------------------------------------------------------------------------------- computed headlines
def _reach(agg: pd.DataFrame, algo: str) -> float | None:
    """Value of x at the first budget (reference backend) whose median gap is within GOOD, else None."""
    g = agg[(agg.algo == algo) & (agg.backend == "reference")].sort_values("budget")
    hit = g[g.y <= GOOD]
    return float(hit.iloc[0]["x"]) if len(hit) else None


def _typical_reach(df: pd.DataFrame, xcol: str) -> dict[str, float] | None:
    """Per algorithm, the median over grid cells of 'x needed to get close enough'; None unless EVERY algorithm gets there
    in EVERY cell (then a one-line comparison would be true everywhere)."""
    d = ok_rows(df)
    d = d[(d.backend == "reference") & d["rel_test_vs_lloyd25_ref"].notna() & d[xcol].notna()]
    algos = [a for a in ALGOS if a in set(d.algo)]
    if len(algos) < 2:
        return None
    per: dict[str, list[float]] = {a: [] for a in algos}
    for n, k in _combos(d):
        agg = _agg(d[(d.n == n) & (d.k == k)], xcol, "rel_test_vs_lloyd25_ref", 100.0)
        for a in algos:
            r = _reach(agg, a)
            if r is None:
                return None
            per[a].append(r)
    return {a: float(np.median(v)) for a, v in per.items()}


def _sentence(vals: dict[str, float], fmt) -> str:
    """'e-means and o-means need 2 passes, k-means needs 10 passes' (fewest first, equal values merged)."""
    groups: dict[float, list[str]] = {}
    for a, v in vals.items():
        groups.setdefault(round(v, 6), []).append(a)
    parts = []
    for v in sorted(groups):
        names = " and ".join(NAME[a] for a in sorted(groups[v], key=ALGOS.index))
        parts.append(f"{names} {'needs' if len(groups[v]) == 1 else 'need'} {fmt(v)}")
    return ", ".join(parts)


def _compact(vals: dict[str, float], fmt, suffix: str = "") -> str:
    """'o-means 1.4, e-means 2, k-means 10 full scans' (fewest first); `fmt` formats the number, `suffix` the shared unit."""
    order = sorted(vals, key=lambda a: (vals[a], ALGOS.index(a)))
    return ", ".join(f"{NAME[a]} {fmt(vals[a])}" for a in order) + suffix


def headline_convergence(df: pd.DataFrame) -> str:
    v = _typical_reach(df, "budget")
    return (f"To get close to k-means' best answer, {_sentence(v, _passes)}" if v
            else "How many passes over the data does each method need?")


def headline_efficiency(df: pd.DataFrame) -> str:
    v = _typical_reach(df, "assign_passes")
    return (f"Work needed to get close to k-means' best answer: {_compact(v, lambda x: f'{x:.2g}', ' full scans')}" if v
            else "How much work does each method need?")


def headline_wallclock(df: pd.DataFrame) -> str:
    v = _typical_reach(df, "t_fit")
    return (f"Time to get close to k-means' best answer (our code): {_compact(v, _secs)}" if v
            else "How long does each method take?")


# ----------------------------------------------------------------------------------------------- chrome
def _wrap(text: str, W: float, pt: float, wide: float = 0.52) -> str:
    return "\n".join(textwrap.wrap(text, max(24, int((W - 1.2) * 72 / (pt * wide)))))


@dataclass
class Layout:
    W: float
    H: float
    title: str
    subtitle: str
    y_title: float
    y_sub: float
    y_leg: float


def _canvas(th: Theme, nrows: int, ncols: int, title: str, subtitle: str, panel=(4.7, 3.5), footer=0.7):
    plt.rcParams.update({"font.family": _font(), "text.color": th.ink, "axes.labelcolor": th.ink2,
                         "xtick.color": th.muted, "ytick.color": th.muted, "figure.dpi": 100})
    pw = max(panel[0], 7.4 / ncols)  # few panels -> wider panels, so the text always fits
    W = pw * ncols + 0.9
    t, s = _wrap(title, W, 15, 0.58), _wrap(subtitle, W, 10)
    tl, sl = t.count("\n") + 1, s.count("\n") + 1
    y_title = 0.28
    y_sub = y_title + 0.30 * tl + 0.10
    y_leg = y_sub + 0.21 * sl + 0.14
    header = y_leg + 0.34 + 0.55  # legend row + room for the panel titles
    H = panel[1] * nrows + header + footer
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, H), squeeze=False, facecolor=th.surface)
    fig.subplots_adjust(left=0.85 / W, right=1 - 0.25 / W, top=1 - header / H, bottom=footer / H + 0.05,
                        wspace=0.3, hspace=0.62)
    return fig, axes, Layout(W, H, t, s, y_title, y_sub, y_leg)


def _style(ax, th: Theme) -> None:
    ax.set_facecolor(th.surface)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th.axis)
        ax.spines[s].set_linewidth(1)
    ax.grid(True, color=th.grid, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color=th.axis, labelsize=9)


def _header(fig, th: Theme, lay: Layout, handles: list, labels: list[str], foot: str) -> None:
    W, H = lay.W, lay.H
    fig.text(0.85 / W, 1 - lay.y_title / H, lay.title, fontsize=15, fontweight="bold", color=th.ink, va="top",
             ha="left", linespacing=1.25)
    fig.text(0.85 / W, 1 - lay.y_sub / H, lay.subtitle, fontsize=10, color=th.ink2, va="top", ha="left", linespacing=1.35)
    if handles:
        leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.78 / W, 1 - lay.y_leg / H),
                         ncol=len(handles), frameon=False, fontsize=9.5, handletextpad=0.5, columnspacing=1.8,
                         handlelength=2.2)
        for t in leg.get_texts():
            t.set_color(th.ink2)
    fig.text(0.85 / W, 0.16 / H, _wrap(foot, W, 8.2), fontsize=8.2, color=th.muted, va="bottom", ha="left",
             linespacing=1.4)


def _handles(th: Theme, algos: list[str], authors: bool = False) -> tuple[list, list[str]]:
    hs = [Line2D([], [], color=th.series[a], lw=2, marker=MARKER[a], markersize=7, markerfacecolor=th.series[a],
                 markeredgecolor=th.surface, markeredgewidth=1.4) for a in algos]
    ls = [LEGEND[a] for a in algos]
    if authors:
        hs.append(Line2D([], [], color=th.muted, lw=0, marker="o", markersize=7, markerfacecolor=th.surface,
                         markeredgecolor=th.muted, markeredgewidth=1.6))
        ls.append("hollow dots: the authors' own code")
    return hs, ls


def _foot(df: pd.DataFrame, spread: bool = True, stretch: bool = True) -> str:
    name = str(df["dataset"].iloc[0]) if len(df) else "?"
    ns = int(df["seed"].nunique())
    parts = [f"{FRIENDLY_DATA.get(name, name)} · {ns} repeat{'s' if ns != 1 else ''}"]
    if spread:
        parts.append("shaded areas show the range across repeats" if ns < 5
                     else "shaded areas show the middle half of the repeats")
    if stretch:
        parts.append("the vertical scale is stretched near 0% so small gaps stay visible")
    return " · ".join(parts)


def _hw(df: pd.DataFrame) -> str:
    try:
        return json.loads(df["hardware_json"].dropna().iloc[0]).get("cpu", "one computer")
    except Exception:
        return "one computer"


def _panel_title(ax, th: Theme, n: int, k: int, prefix: str = "") -> None:
    ax.set_title(f"{prefix}{n:,} points · {k:,} groups", loc="left", fontsize=10, color=th.ink2, pad=9)


# ----------------------------------------------------------------------------------------------- axes
def _gap_axis(ax, th: Theme, which: str = "y", direction: bool = True, linthresh: float = 0.5) -> None:
    """The 'gap to k-means' best' axis: stretched near 0%, with worse/better cues."""
    getattr(ax, f"set_{which}scale")("symlog", linthresh=linthresh, linscale=0.9)
    lim = ax.get_ylim() if which == "y" else ax.get_xlim()
    lo = lim[0] * 1.3 if lim[0] < 0 else lim[0]
    hi = lim[1] * 1.3 if lim[1] > 0 else lim[1]
    pad = (min(lo, -0.8), max(hi, 0.8))
    (ax.set_ylim if which == "y" else ax.set_xlim)(pad)
    cand = [-5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100]
    ticks = [t for t in cand if pad[0] - 1e-9 <= t <= pad[1] + 1e-9]
    ax_ = ax.yaxis if which == "y" else ax.xaxis
    ax_.set_major_locator(FixedLocator(ticks))
    ax_.set_major_formatter(FuncFormatter(lambda v, _: "0%" if v == 0 else f"{v:+g}%"))
    ax_.set_minor_formatter(NullFormatter())
    if direction and which == "y":
        ax.annotate("worse", xy=(0.992, 0.985), xycoords="axes fraction", va="top", ha="right", fontsize=8.5, color=th.muted)
        ax.annotate("better", xy=(0.992, 0.015), xycoords="axes fraction", va="bottom", ha="right", fontsize=8.5,
                    color=th.muted)


def _close_zone(ax, th: Theme, label: bool = True) -> None:
    """Shaded 'close enough' band around the k-means result, plus its zero line."""
    ax.axhspan(-GOOD, GOOD, color=th.axis, alpha=0.30, linewidth=0, zorder=0)
    ax.axhline(0, color=th.axis, linewidth=1.2, zorder=1)
    if label:
        ax.annotate("close enough", xy=(0.995, GOOD), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8.2, color=th.muted)


def _log_axis(ax, ticks: list[float], which: str = "x", fmt=None) -> None:
    getattr(ax, f"set_{which}scale")("log")
    ax_ = ax.xaxis if which == "x" else ax.yaxis
    ax_.set_major_locator(FixedLocator(ticks))
    ax_.set_major_formatter(FuncFormatter(fmt or (lambda v, _: _fmt_num(v) if v >= 1 else f"{v:g}")))
    ax_.set_minor_formatter(NullFormatter())


def _plot_series(ax, th: Theme, agg: pd.DataFrame, backend: str, joined_official: bool = False) -> None:
    for algo in ALGOS:
        g = agg[(agg.algo == algo) & (agg.backend == backend)].sort_values("x")
        if g.empty:
            continue
        c = th.series[algo]
        if backend == "reference":
            ax.fill_between(g["x"], g["lo"], g["hi"], color=c, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(g["x"], g["y"], color=c, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            ax.plot(g["x"], g["y"], linestyle="", marker=MARKER[algo], markersize=7, markerfacecolor=c,
                    markeredgecolor=th.surface, markeredgewidth=1.4, zorder=4)
        else:  # the authors' own code: hollow markers (a check on our code)
            if joined_official:
                ax.plot(g["x"], g["y"], color=c, lw=2, zorder=3)
            ax.plot(g["x"], g["y"], linestyle="", marker=MARKER[algo], markersize=8, markerfacecolor=th.surface,
                    markeredgecolor=c, markeredgewidth=1.8, zorder=5)


def _grid_layout(combos: list, ncols_max: int = 3) -> tuple[int, int]:
    ncols = min(ncols_max, len(combos))
    return math.ceil(len(combos) / ncols), ncols


def _ylab(ax, i: int, ncols: int, text: str = "Gap to k-means' best answer") -> None:
    if i % ncols == 0:
        ax.set_ylabel(text, fontsize=9.5)


# ----------------------------------------------------------------------------------------------- figures
def fig_convergence(df: pd.DataFrame, th: Theme, exp: str):
    """How many passes over the data does each method need to get close to k-means' best answer?"""
    d = ok_rows(df)
    d = d[d["rel_test_vs_lloyd25_ref"].notna()]
    if d.empty:
        return None
    combos = _combos(d)
    nrows, ncols = _grid_layout(combos)
    algos = [a for a in ALGOS if a in set(d.algo)]
    authors = bool((d.backend == "official").any())
    fig, axes, lay = _canvas(th, nrows, ncols, headline_convergence(d),
                             f"Lower is better. The shaded zone is close enough: {CLOSE}.")
    for ax in axes.ravel():
        ax.set_visible(False)
    out = []
    for i, (n, k) in enumerate(combos):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        _style(ax, th)
        sub = d[(d.n == n) & (d.k == k)]
        agg = _agg(sub, "budget", "rel_test_vs_lloyd25_ref", 100.0)
        _close_zone(ax, th)
        for backend in ("reference", "official"):
            _plot_series(ax, th, agg, backend)
        out.append(agg.assign(n=n, k=k)[["n", "k", *agg.columns]])
        xs = sorted(set(sub.budget))
        _log_axis(ax, [t for t in (1, 2, 3, 5, 10, 25) if xs[0] / 1.3 <= t <= xs[-1] * 1.3] or xs[:1])
        ax.set_xlim(min(xs) / 1.25, max(xs) * 1.25)
        _gap_axis(ax, th)
        _panel_title(ax, th, n, k)
        ax.set_xlabel("Passes over the data", fontsize=9.5)
        _ylab(ax, i, ncols)
    hs, ls = _handles(th, algos, authors)
    _header(fig, th, lay, hs, ls, _foot(d))
    return fig, pd.concat(out, ignore_index=True)


def fig_efficiency(df: pd.DataFrame, th: Theme, exp: str):
    """Quality against the amount of work done (counted, so it does not depend on the computer)."""
    d = ok_rows(df)
    d = d[d["rel_test_vs_lloyd25_ref"].notna() & (d["backend"] == "reference")]
    if d.empty:
        return None
    combos = _combos(d)
    nrows, ncols = _grid_layout(combos)
    algos = [a for a in ALGOS if a in set(d.algo)]
    fig, axes, lay = _canvas(th, nrows, ncols, headline_efficiency(d),
                             f"Work is counted, not timed, so it does not depend on the computer. Shaded zone: {CLOSE}.")
    for ax in axes.ravel():
        ax.set_visible(False)
    out = []
    for i, (n, k) in enumerate(combos):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        _style(ax, th)
        sub = d[(d.n == n) & (d.k == k)]
        agg = _agg(sub, "assign_passes", "rel_test_vs_lloyd25_ref", 100.0)
        _close_zone(ax, th)
        _plot_series(ax, th, agg, "reference")
        out.append(agg.assign(n=n, k=k)[["n", "k", *agg.columns]])
        xs = [v for v in agg["x"] if v > 0]
        lo, hi = min(xs) / 1.3, max(xs) * 1.3
        _log_axis(ax, [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 25) if lo <= t <= hi])
        ax.set_xlim(lo, hi)
        _gap_axis(ax, th)
        _panel_title(ax, th, n, k)
        ax.set_xlabel("Amount of work (1 = one full scan of the data)", fontsize=9.5)
        _ylab(ax, i, ncols)
    hs, ls = _handles(th, algos)
    _header(fig, th, lay, hs, ls, _foot(d))
    return fig, pd.concat(out, ignore_index=True)


def fig_wallclock(df: pd.DataFrame, th: Theme, exp: str):
    """Quality against time, one panel row per implementation: only compare lines inside a row."""
    d = ok_rows(df)
    d = d[d["rel_test_vs_lloyd25_ref"].notna() & d["t_fit"].notna()]
    if d.empty:
        return None
    combos = _combos(d)
    backends = [b for b in ("reference", "official") if b in set(d.backend)]
    per_row, ncols = _grid_layout(combos)
    algos = [a for a in ALGOS if a in set(d.algo)]
    fig, axes, lay = _canvas(
        th, len(backends) * per_row, ncols, headline_wallclock(d),
        f"Compare lines only within a row: the two rows ran different software. Shaded zone: {CLOSE}. "
        "o-means is built for GPUs, so its time on this CPU is not representative.")
    for ax in axes.ravel():
        ax.set_visible(False)
    out = []
    for bi, b in enumerate(backends):
        for i, (n, k) in enumerate(combos):
            ax = axes[bi * per_row + i // ncols, i % ncols]
            sub = d[(d.n == n) & (d.k == k) & (d.backend == b)]
            if sub.empty:
                continue
            ax.set_visible(True)
            _style(ax, th)
            agg = _agg(sub, "t_fit", "rel_test_vs_lloyd25_ref", 100.0)
            _close_zone(ax, th, label=(bi == 0))
            _plot_series(ax, th, agg, b, joined_official=True)
            out.append(agg.assign(n=n, k=k)[["n", "k", *agg.columns]])
            xs = [v for v in agg["x"] if v > 0]
            lo, hi = min(xs) / 1.6, max(xs) * 1.6
            ticks = [t for t in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300) if lo <= t <= hi]
            _log_axis(ax, ticks or [min(xs)], fmt=lambda v, _: f"{v:g}")
            ax.set_xlim(lo, hi)
            _gap_axis(ax, th)
            _panel_title(ax, th, n, k, prefix="Our code · " if b == "reference" else "Authors' code (Faiss) · ")
            ax.set_xlabel("Time to train (seconds)", fontsize=9.5)
            _ylab(ax, i, ncols)
    hs, ls = _handles(th, algos)
    _header(fig, th, lay, hs, ls, _foot(d))
    return fig, pd.concat(out, ignore_index=True)


def fig_train_vs_test(df: pd.DataFrame, th: Theme, exp: str):
    """Does the quality on the data a method learned from carry over to new data?"""
    d = ok_rows(df)
    d = d[d["rel_test_vs_lloyd25_ref"].notna() & d["rel_train_vs_lloyd25_ref"].notna() & (d["backend"] == "reference")]
    if d.empty:
        return None
    combos = _combos(d)
    nrows, ncols = _grid_layout(combos)
    algos = [a for a in ALGOS if a in set(d.algo)]
    fig, axes, lay = _canvas(th, nrows, ncols, "Does it also work on data the method has never seen?",
                             "Each dot is one setting. A dot on the diagonal means equally good on the data it learned "
                             "from and on new data; below it, better on new data.")
    for ax in axes.ravel():
        ax.set_visible(False)
    out = []
    for i, (n, k) in enumerate(combos):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        _style(ax, th)
        sub = d[(d.n == n) & (d.k == k)]
        rows = []
        for (algo, budget), g in sub.groupby(["algo", "budget"]):
            rows.append({"algo": algo, "budget": budget, "train": 100 * g.rel_train_vs_lloyd25_ref.median(),
                         "test": 100 * g.rel_test_vs_lloyd25_ref.median()})
        t = pd.DataFrame(rows)
        lim = [min(t.train.min(), t.test.min(), -0.2) * 1.15, max(t.train.max(), t.test.max(), 0.2) * 1.15]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.plot(lim, lim, color=th.axis, lw=1.2, zorder=1)
        for algo in algos:
            g = t[t.algo == algo].sort_values("budget")
            c = th.series[algo]
            ax.plot(g.train, g.test, color=c, lw=2, zorder=3)
            ax.plot(g.train, g.test, linestyle="", marker=MARKER[algo], markersize=7, markerfacecolor=c,
                    markeredgecolor=th.surface, markeredgewidth=1.4, zorder=4)
            dx, dy, ha = LABEL_OFFSET[algo]  # each algorithm labels in its own direction so labels never collide
            for _, r in pd.concat([g.head(1), g.tail(1)]).drop_duplicates().iterrows():  # label the two extremes only
                ax.annotate(_passes(r.budget), (r.train, r.test), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, fontsize=8.2, color=th.ink2)
        _gap_axis(ax, th, "x", direction=False)
        _gap_axis(ax, th, "y", direction=False)
        ax.axhline(0, color=th.axis, lw=0.8, zorder=1)
        ax.axvline(0, color=th.axis, lw=0.8, zorder=1)
        _panel_title(ax, th, n, k)
        ax.set_xlabel("Gap on the data it learned from", fontsize=9.5)
        _ylab(ax, i, ncols, "Gap on new data")
        out.append(t.assign(n=n, k=k)[["n", "k", *t.columns]])
    hs, ls = _handles(th, algos)
    _header(fig, th, lay, hs, ls,
            _foot(d, spread=False) + " · labels show the fewest and the most passes tried")
    return fig, pd.concat(out, ignore_index=True)


def fig_omeans_cost(df: pd.DataFrame, th: Theme, exp: str):
    """How much work does o-means actually skip, compared with e-means?"""
    d = ok_rows(df)
    d = d[d["backend"] == "reference"]
    o, e = d[d.algo == "omeans"], d[d.algo == "emeans"]
    if o.empty or e.empty:
        return None
    m = o.merge(e[["n", "k", "seed", "budget", "distance_computations"]], on=["n", "k", "seed", "budget"],
                suffixes=("", "_e"))
    if m.empty:
        return None
    m["total_vs_e"] = m["distance_computations"] / m["distance_computations_e"]
    combos = _combos(m)
    nrows, ncols = _grid_layout(combos)
    typical = float(m.groupby(["n", "k", "budget"]).total_vs_e.median().median())
    title = (f"o-means does about {typical:.0%} of the work e-means does" if typical < 0.98
             else "o-means does about as much work as e-means")
    fig, axes, lay = _canvas(th, nrows, ncols, title,
                             "Filled: o-means' total work compared with e-means, at the same number of passes. Hollow: how "
                             "much of a full search it still performs. Lower means more work skipped.")
    for ax in axes.ravel():
        ax.set_visible(False)
    c = th.series["omeans"]
    out = []
    for i, (n, k) in enumerate(combos):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        _style(ax, th)
        sub = m[(m.n == n) & (m.k == k)]
        g = sub.groupby("budget").agg(total=("total_vs_e", "median"), actual=("assign_fraction_actual", "median"),
                                      nominal=("assign_fraction_nominal", "median")).reset_index()
        x = np.arange(len(g))
        nom = float(g.nominal.median())
        ax.axhline(nom, color=th.axis, lw=1.2, zorder=1)
        ax.annotate("what the design promises on paper", xy=(1.0, nom), xycoords=("axes fraction", "data"), xytext=(-2, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8.2, color=th.muted)
        ax.plot(x, g.total, color=c, lw=2, zorder=3)
        ax.plot(x, g.total, linestyle="", marker="^", markersize=8, markerfacecolor=c, markeredgecolor=th.surface,
                markeredgewidth=1.4, zorder=4)
        ax.plot(x, g.actual, linestyle="", marker="^", markersize=8, markerfacecolor=th.surface, markeredgecolor=c,
                markeredgewidth=1.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_fmt_num(b) for b in g.budget])
        ax.set_xlim(-0.5, len(g) - 0.5)
        ax.set_ylim(0, max(1.05, float(g[["total", "actual"]].max().max()) * 1.1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        _panel_title(ax, th, n, k)
        ax.set_xlabel("Passes over the data", fontsize=9.5)
        _ylab(ax, i, ncols, "Share of the work")
        out.append(g.assign(n=n, k=k)[["n", "k", *g.columns]])
    hs = [Line2D([], [], color=c, lw=2, marker="^", markersize=8, markerfacecolor=c, markeredgecolor=th.surface,
                 markeredgewidth=1.4),
          Line2D([], [], color=c, lw=0, marker="^", markersize=8, markerfacecolor=th.surface, markeredgecolor=c,
                 markeredgewidth=1.8)]
    _header(fig, th, lay, hs, ["Total work vs e-means (including keeping its index up to date)",
                               "Share of a full search it still performs"],
            _foot(m, spread=False, stretch=False) + " · medians across repeats")
    return fig, pd.concat(out, ignore_index=True)


def fig_parity(df: pd.DataFrame, th: Theme, exp: str):
    """Does our code give the same answers as the authors' own code?"""
    d = ok_rows(df)
    keys = ["n", "k", "seed", "algo", "budget"]
    ref = d[(d.backend == "reference") & d.algo.isin(["lloyd", "emeans"])]
    off = d[(d.backend == "official") & d.algo.isin(["lloyd", "emeans"])]
    if ref.empty or off.empty:
        return None
    m = ref.merge(off[keys + ["test_mse", "train_mse"]], on=keys, suffixes=("", "_off"))
    if m.empty:
        return None
    m["diff_test_pct"] = 100 * (m.test_mse - m.test_mse_off) / m.test_mse_off
    m["diff_train_pct"] = 100 * (m.train_mse - m.train_mse_off) / m.train_mse_off
    worst = float(m.diff_test_pct.abs().max())
    title = (f"Our code matches the authors' code to within {worst:.2g}%" if worst <= GOOD else
             f"Our code differs from the authors' code by up to {worst:.2g}%, more than the {GOOD:g}% we count as the same")
    combos = _combos(m)
    nrows, ncols = _grid_layout(combos)
    fig, axes, lay = _canvas(th, nrows, ncols, title,
                             f"Each dot is one run with the same data and the same starting point. The shaded zone "
                             f"(±{GOOD:g}%) is counted as the same result.", panel=(4.7, 3.3))
    for ax in axes.ravel():
        ax.set_visible(False)
    algos = [a for a in ("lloyd", "emeans") if a in set(m.algo)]
    out = []
    rng = np.random.default_rng(0)
    for i, (n, k) in enumerate(combos):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        _style(ax, th)
        sub = m[(m.n == n) & (m.k == k)]
        slots = sorted({(a, b) for a, b in zip(sub.algo, sub.budget)}, key=lambda t: (ALGOS.index(t[0]), t[1]))
        ax.axhspan(-GOOD, GOOD, color=th.axis, alpha=0.30, linewidth=0, zorder=0)
        ax.axhline(0, color=th.axis, lw=1.2, zorder=1)
        for xi, (algo, budget) in enumerate(slots):
            g = sub[(sub.algo == algo) & (sub.budget == budget)]
            c = th.series[algo]
            ax.plot(xi + rng.uniform(-0.12, 0.12, len(g)), g.diff_test_pct, linestyle="", marker=MARKER[algo],
                    markersize=7, markerfacecolor=c, markeredgecolor=th.surface, markeredgewidth=1.2, alpha=0.6, zorder=3)
            med = float(g.diff_test_pct.median())
            ax.plot([xi - 0.3, xi + 0.3], [med, med], color=c, lw=2.4, zorder=4)
        ax.set_xticks(range(len(slots)))
        ax.set_xticklabels([f"{NAME[a]}\n{_passes(b)}" for a, b in slots], fontsize=8.8)
        ax.set_xlim(-0.6, len(slots) - 0.4)
        top = max(0.8, float(m.diff_test_pct.abs().max()) * 1.25)
        ax.set_ylim(-top, top)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0%" if v == 0 else f"{v:+g}%"))
        ax.annotate("ours is worse", xy=(0.992, 0.985), xycoords="axes fraction", va="top", ha="right", fontsize=8.5, color=th.muted)
        ax.annotate("ours is better", xy=(0.992, 0.015), xycoords="axes fraction", va="bottom", ha="right", fontsize=8.5,
                    color=th.muted)
        _panel_title(ax, th, n, k)
        _ylab(ax, i, ncols, "Our code vs. authors' code")
        out.append(sub[keys + ["test_mse", "test_mse_off", "diff_test_pct", "train_mse", "train_mse_off", "diff_train_pct"]])
    hs, ls = _handles(th, algos)
    _header(fig, th, lay, hs, ls, _foot(m, spread=False, stretch=False) + " · the short bar marks the middle of the dots")
    return fig, pd.concat(out, ignore_index=True)


FIGURES = {
    "convergence": fig_convergence,
    "efficiency": fig_efficiency,
    "wallclock": fig_wallclock,
    "train_vs_test": fig_train_vs_test,
    "omeans_cost": fig_omeans_cost,
    "parity": fig_parity,
}


def make_figures(df: pd.DataFrame, out_dir: str | Path, experiment: str, themes: tuple[str, ...] = ("light",),
                 only: tuple[str, ...] | None = None) -> dict[str, list[Path]]:
    """Render every applicable figure for the requested themes; returns {figure name: [files]} (skipped figures are absent)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        for tname in themes:
            res = fn(df, THEMES[tname], experiment)
            if res is None:
                break
            fig, table = res
            suffix = "" if tname == "light" else f"_{tname}"
            png = out_dir / f"{name}{suffix}.png"
            fig.savefig(png, dpi=160, facecolor=fig.get_facecolor())
            plt.close(fig)
            files = [png]
            if tname == "light":  # the table twin does not depend on the theme
                csv = out_dir / f"{name}.csv"
                table.to_csv(csv, index=False)
                files.append(csv)
            written.setdefault(name, []).extend(files)
    return written
