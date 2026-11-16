<p align="center">
  <img src="https://raw.githubusercontent.com/rajveer43/keo/main/assets/logo.svg" alt="KEO logo: a ninja cat above three clusters of dots, next to the letters K, E, O" width="420">
</p>

# KEO

**A reproducible benchmark of k-means, e-means and o-means on data neither method was evaluated on.**

**What KEO stands for.** KEO is named after Keo, the ninja cat from the *Ninja Hattori* cartoon series, and it spells out the three clustering algorithms the framework compares: **K**-means, **E**-means and **O**-means. In the logo, the masked ninja cat stands over three clusters of points, each with its centroid ringed, and the letters K, E and O use one colour per algorithm, the same colours as every figure.

KEO compares three large-scale clustering algorithms under one protocol, one set of metrics and one result store, and tests whether the headline claims of the e-means and o-means papers hold outside the datasets they were developed on.

| | Algorithm | Summary | Source |
|---|---|---|---|
| **K** | k-means (Lloyd) | Full-batch alternation of exact assignment and centroid mean update. | Lloyd (1982) |
| **E** | e-means | Mini-batch k-means with exponential-moving-average updates, a cosine step schedule, scaling laws for batch size and step size, and dead-cluster revival. Reported to match Lloyd in a fraction of the epochs. | Bessa et al., Meta FAIR, 2026 |
| **O** | o-means | e-means with a two-level inverted-file index built over the centroids, so each point is compared with a small subset of them instead of all `k`. | Bajaj, Nupur, Garg, 2026 |

## Contents

- [Research question](#research-question)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Method](#method)
- [Project status](#project-status)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [References](#references)

## Research question

Do the reported gains of e-means (matching k-means quality in roughly 1.4–2.6 epochs) and o-means (further reducing assignment cost as `k` grows) hold on a dataset that was not used to design or tune them, and under what conditions do they break down?

## Dataset

**GloVe-100**: 1,183,514 word vectors in 100 dimensions, with 10,000 provided query vectors, taken from ANN-Benchmarks (`glove-100-angular`). It appears in neither paper's list of evaluation datasets (checked against those two papers only). Vectors are L2-normalised by default. The download is checksum-verified (SHA-256).

## Installation

Requires Python 3.10 or newer (developed and tested on 3.12).

```bash
pip install keo
```

To work on the code, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/rajveer43/keo.git
cd keo
uv sync
```

## Quick start

```bash
keo download glove100     # ~460 MB, checksum-verified
keo smoke                 # end-to-end run on a small subset, about 10 seconds
keo figures smoke         # figures in results/smoke/figures/
keo adapters              # which official implementations can run on this machine
```

From a source checkout, prefix commands with `uv run` and run the tests with `uv run pytest`.

Where files go: in a source checkout, `data/` and `results/` sit next to the code. After `pip install`, datasets are cached in `~/.cache/keo/data` and results are written to `./results` in the directory you run from. Set `KEO_DATA_DIR`, `KEO_RESULTS_DIR` or `KEO_CONFIGS_DIR` to override.

Run a configured experiment, or preview it first:

```bash
keo run <name-or-path.yaml> --dry-run
keo run <name-or-path.yaml>
keo status <name>
keo figures <name> --theme both
```

Each result cell is stored under a content hash of its configuration, so an interrupted run resumes where it stopped and completed cells are never recomputed.

## Method

**Two implementations, never mixed.** Every algorithm exists twice:

- `reference`: our own PyTorch (CPU) implementation, written from the papers.
- `official`: the authors' released code (Faiss `contrib/e_means` and related adapters), run in an isolated worker process.

Each result row records which one produced it. Quality is compared across both; wall-clock time is compared only within one implementation.

**Metrics.**

- MSE: mean squared L2 distance to the assigned centroid, computed with exact assignment in float64, on both training and held-out points.
- Relative MSE (the *gap*): error relative to k-means run for 25 iterations from the same starting centroids and seed.
- Cluster health: dead clusters (fewer than 10 points), empty clusters, and the Faiss imbalance factor.
- Cost: counted distance computations, split into assignment and index maintenance, reported both as actually performed and as the nominal cell arithmetic. This is hardware independent.
- Wall-clock fit time, within one implementation only.

**e-means recipe.** `recipe="official"` (default) uses the constants shipped in the official code; `recipe="paper"` uses the values in the paper text. The two differ slightly. All headline results use `official`.

**Figures.** Six figures are generated from the stored result rows, each with a CSV of exactly the plotted numbers and a light and dark variant. Headlines are computed from the data rather than written by hand. See [docs/FIGURES.md](docs/FIGURES.md).

## Project status

| Phase | Scope | State |
|---|---|---|
| 0–2 | Specification, data pipeline, metrics, k-means | Complete |
| 3–4 | e-means and o-means reference implementations | Complete |
| 5 | Adapters for the official implementations | Complete |
| 6 | Result store, resumable runner, figures | Complete |
| 7 | Main run on the full grid | Not started |
| 8 | Analysis and report (`keo analyze`, `keo report`) | Not started; commands are stubs |
| 9 | Ablations, scaling-law transfer, secondary track | Not started |
| 10 | Hardening and packaging | Not started |

The test suite has 179 tests, all passing. The `smoke` experiment (26 cells) runs cleanly end to end.

**Implementation agreement.** On GloVe subsets, our e-means matches the official implementation within about 0.2% on held-out error. The remainder is most likely remainder-batch handling (see assumptions A2 and A52 in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)); this has not been isolated.

**No benchmark results are claimed yet.** Numbers visible in `results/smoke/` come from two repeats at `k = 512` and exist to verify the pipeline, not to answer the research question.

## Limitations

- **o-means is partly reconstructed.** Its paper is unreleased, so our version is built from the project page and every unspecified detail is a recorded assumption. Conclusions about o-means describe this reconstruction.
- **No GPU in the reference environment.** Wall-clock claims from the papers concern GPU execution and cannot be tested on the development machine (Apple M4, CPU only). Quality and operation counts can be.
- **CPU time is not representative for o-means.** Index maintenance is cheap on a GPU and relatively costly on a CPU, so CPU wall-clock should not be read as a verdict on the method.
- **One dataset so far.** Findings are about GloVe-100 and will not necessarily transfer.
- **Dataset novelty is a bounded claim.** GloVe-100 was checked against the two papers' dataset lists, not against all prior work.

## License

Code is released under the [MIT License](LICENSE). The GloVe vectors are not redistributed; they are downloaded from ANN-Benchmarks and remain under their own terms.

## Documentation

| File | Contents |
|---|---|
| [docs/SPEC.md](docs/SPEC.md) | Algorithms, formulas and experimental protocol, tied to the source papers |
| [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | Register of every point where the papers were silent, and the decision taken |
| [docs/FIGURES.md](docs/FIGURES.md) | What each figure answers, how it is built, and the palette validation record |
| [docs/MASTER_PROMPT.md](docs/MASTER_PROMPT.md) | The original build brief (archived; has since diverged) |

`docs/` is excluded from version control (see `.gitignore`) and is kept locally.

## Repository layout

```
configs/         Dataset and experiment definitions (YAML)
src/keo/
  algos/         k-means, e-means, o-means, scaling laws, revival, IVF cells; adapters/ for official code
  bench/         Experiment schema, resumable runner, result store, provenance
  metrics/       MSE, gap, cluster health, operation counting
  analysis/      Figure generation
  data/          Download, verification, preparation
tests/           Unit tests and per-phase gate tests
results/         Run outputs (smoke, exploratory)
docs/            Specification, assumptions, figure guide (local only)
```

## References

- **e-means**: Bessa et al., *e-means: outperforming k-means (in one-tenth the epochs)*. [arXiv:2611.09501](https://arxiv.org/abs/2611.09501) · [project page](https://e-means.github.io) · code in Faiss `contrib/e_means`
- **o-means**: Bajaj, Nupur, Garg, *o-means: index the centroids, don't scan them*. Project page, November 2026; paper not yet released.
- **Flash-KMeans** (GPU k-means baseline): [arXiv:2603.09229](https://arxiv.org/abs/2603.09229)
- **Faiss**: Douze et al., *The Faiss library*. [arXiv:2401.08281](https://arxiv.org/abs/2401.08281)
- **VQ-VAE** (origin of the moving-average update): van den Oord et al. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- **ANN-Benchmarks**: Aumüller et al. [arXiv:1807.05614](https://arxiv.org/abs/1807.05614) · **GloVe**: Pennington et al. [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) · **VIBE**: [arXiv:2505.17810](https://arxiv.org/abs/2505.17810)
- **Classics**: Lloyd (1982); Arthur and Vassilvitskii, *k-means++* (2007); Sculley, *Web-scale k-means clustering* (2010)
