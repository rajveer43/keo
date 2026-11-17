# KEO

<p align="center">
  <img src="https://raw.githubusercontent.com/rajveer43/keo/main/assets/logo.svg" alt="KEO logo" width="360">
</p>

<p align="center">
  <a href="https://pypi.org/project/keo/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/keo?logo=pypi&logoColor=white"></a>
  <a href="https://pypi.org/project/keo/"><img alt="Monthly downloads" src="https://img.shields.io/pypi/dm/keo?logo=pypi&logoColor=white"></a>
  <a href="https://pypi.org/project/keo/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/keo?logo=python&logoColor=white"></a>
  <a href="https://github.com/rajveer43/keo/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/rajveer43/keo/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/rajveer43/keo/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/keo"></a>
  <a href="https://doi.org/10.5281/zenodo.24273473"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.24273473.svg"></a>
  <img alt="Developed on Apple Silicon" src="https://img.shields.io/badge/developed%20on-Apple%20Silicon-lightgrey?logo=apple&logoColor=white">
</p>

**A reproducible evaluation of k-means, e-means, and o-means on an independent dataset.**

KEO is an experimental benchmarking framework for evaluating three large-scale clustering algorithms under a unified experimental protocol. It provides consistent implementations, evaluation metrics, experiment tracking, and result generation to investigate whether the performance improvements reported by recent e-means and o-means research generalize to datasets outside their original evaluations.

The goal is not to establish another leaderboard, but to conduct a controlled and reproducible investigation of algorithmic performance, computational cost, and practical limitations.

## Research motivation

Recent work has proposed improvements to traditional k-means clustering:

- **k-means:** The classical Lloyd algorithm, serving as the baseline.
- **e-means:** A mini-batch approach using exponential moving average updates and adaptive optimization strategies, reported to achieve comparable clustering quality in substantially fewer epochs.
- **o-means:** An extension of e-means that uses an inverted-file index over centroids to reduce assignment costs as the number of clusters increases.

KEO evaluates these approaches using the same dataset, initialization, evaluation metrics, and experiment-tracking infrastructure.

| Algorithm | Approach | Reference |
|---|---|---|
| **K — k-means** | Lloyd's algorithm with exact assignment and centroid updates. | Lloyd (1982) |
| **E — e-means** | Mini-batch k-means with EMA updates, cosine step scheduling, scaling laws, and dead-cluster revival. | Bessa et al. (2026) |
| **O — o-means** | e-means with a two-level inverted-file index over centroids to reduce assignment comparisons. | Bajaj, Nupur, Garg (2026) |

## Research question

The central research question is:

**Do the reported improvements of e-means and o-means generalize to datasets that were not used to develop or tune these algorithms?**

Specifically, KEO investigates:

- Whether e-means can achieve k-means-level clustering quality in approximately 1.4–2.6 epochs.
- Whether o-means reduces assignment costs as the number of clusters increases.
- How clustering quality, computational cost, and cluster stability change across experimental configurations.
- Under what conditions these algorithms fail to deliver their expected improvements.

## Dataset

### GloVe-100

KEO currently uses the GloVe-100 dataset from [ANN-Benchmarks](https://ann-benchmarks.com/).

| Property | Value |
|---|---|
| Dataset | GloVe-100 |
| Number of vectors | 1,183,514 |
| Dimensions | 100 |
| Query vectors | 10,000 |
| Source | ANN-Benchmarks |
| Configuration | `glove-100-angular` |
| Preprocessing | L2 normalization |
| Integrity verification | SHA-256 checksum |

The dataset was selected because it does not appear in the evaluation dataset lists of either the e-means or o-means papers, based on the papers and project materials examined for this study.

The dataset is downloaded automatically and verified using its SHA-256 checksum.

## Installation

### Requirements

- Python 3.10 or newer
- Python 3.12 recommended for the development environment

Install KEO using pip:

```bash
pip install keo
```

### Development installation

Clone the repository and install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/rajveer43/keo.git
cd keo

uv sync
```

## Quick start

Run your first experiment:

```bash
# Download the GloVe-100 dataset (~460 MB)
keo download glove100

# Run an end-to-end smoke test
keo smoke

# Generate figures from the smoke experiment
keo figures smoke

# Check available official implementations
keo adapters
```

The smoke experiment runs on a small subset of the dataset and takes approximately 10 seconds.

### Running experiments

Preview an experiment before execution:

```bash
keo run <name-or-path.yaml> --dry-run
```

Execute the experiment:

```bash
keo run <name-or-path.yaml>
```

Inspect experiment progress:

```bash
keo status <name>
```

Generate figures:

```bash
keo figures <name> --theme both
```

From a source checkout, prefix commands with `uv run`. For example:

```bash
uv run keo smoke
uv run pytest
```

### Data and results

KEO uses configurable directories for datasets, results, and experiment configurations.

| Environment variable | Purpose |
|---|---|
| `KEO_DATA_DIR` | Dataset storage |
| `KEO_RESULTS_DIR` | Experiment results |
| `KEO_CONFIGS_DIR` | Experiment configurations |

By default:

- Source installations use `data/` and `results/` within the repository.
- Installed packages cache datasets in `~/.cache/keo/data`.
- Results are written to `./results` relative to the current working directory.

## Methodology

### Implementation strategy

Each algorithm is evaluated through two separate implementations.

**1. Reference implementation**

Our own PyTorch CPU implementations, developed from the corresponding research papers.

**2. Official implementation**

The authors' released implementations, accessed through isolated worker processes and adapters, including Faiss `contrib/e_means`.

These implementations are never mixed within an experiment. Every result records its implementation provenance.

Clustering quality is compared across implementations. Wall-clock execution time is compared only within the same implementation to avoid misleading performance comparisons.

### Evaluation metrics

KEO tracks the following metrics:

| Metric | Description |
|---|---|
| MSE | Mean squared L2 distance to the assigned centroid, evaluated on training and held-out points. |
| Relative MSE (Gap) | Error relative to k-means run for 25 iterations from the same initialization and random seed. |
| Cluster health | Dead clusters, empty clusters, and Faiss imbalance factor. |
| Computational cost | Distance computations for assignment and index maintenance. |
| Wall-clock time | Fitting time, compared within the same implementation. |

MSE is computed using exact assignment in float64 precision on both training and held-out points.

Computational cost is reported in two forms:

- Actual distance computations performed.
- Nominal cell arithmetic.

This provides a hardware-independent perspective on algorithmic complexity.

### e-means configurations

KEO supports two e-means recipes:

- `official`: Uses the constants provided by the official implementation. This is the default configuration.
- `paper`: Uses the values specified in the research paper.

The two configurations differ slightly. All headline evaluations use the official recipe.

### Reproducibility and experiment tracking

Reproducibility is a central design principle of KEO.

Each experiment result is associated with a content hash derived from its configuration.

This provides several benefits:

- Experiments can resume after interruption.
- Previously completed result cells are not recomputed.
- Configuration provenance is preserved.
- Results remain traceable to their corresponding experimental settings.

### Figures and reporting

KEO generates six figures directly from stored experiment results.

Each figure includes:

- A corresponding CSV containing the exact plotted values.
- Light and dark theme variants.
- Data-driven headline calculations.

No headline metrics are manually entered into the figures.

## Project status

KEO is under active development. The experimental infrastructure and core algorithm implementations are complete, while the main benchmark and analysis phases remain in progress.

| Phase | Scope | Status |
|---|---|---|
| 0–2 | Specification, dataset pipeline, metrics, k-means | Complete |
| 3–4 | e-means and o-means reference implementations | Complete |
| 5 | Official implementation adapters | Complete |
| 6 | Result storage, resumable execution, figure generation | Complete |
| 7 | Full benchmark grid | Not started |
| 8 | Analysis and report generation | Not started |
| 9 | Ablations, scaling-law transfer, secondary experiments | Not started |
| 10 | Hardening and packaging | Not started |

### Current validation

- 179 automated tests passing on the development machine.
- Smoke experiment covering 26 experimental cells.
- Successful end-to-end execution of the smoke pipeline.
- GitHub Actions CI badge tracks the latest Linux test run.

### Implementation agreement

On GloVe subsets, the reference e-means implementation matches the official implementation within approximately 0.2% on held-out error.

The remaining difference is suspected to be related to remainder-batch handling, but this has not yet been isolated.

**Important: No benchmark conclusions are claimed yet.**

Results currently available in `results/smoke/` are based on two repetitions at `k = 512`. They validate the experimental pipeline and should not be interpreted as answers to the research question.

## Limitations

The current evaluation has several limitations that should be considered when interpreting its results.

**1. o-means reconstruction**

The o-means paper has not been released. Its implementation is reconstructed using the available project page, with unspecified details documented as explicit assumptions. Conclusions about o-means therefore apply to this reconstruction.

**2. Hardware limitations**

The reference development environment uses an Apple M4 CPU without GPU execution for the reference implementations. GPU performance claims from the original papers cannot currently be reproduced in this environment.

Clustering quality and computational operation counts can still be evaluated.

**3. CPU timing limitations**

Index maintenance in o-means may have different performance characteristics on GPUs and CPUs. Consequently, CPU wall-clock measurements should not be treated as a definitive assessment of the algorithm's performance.

**4. Dataset coverage**

The current evaluation uses only GloVe-100. Results may not generalize to other datasets, dimensionalities, or data distributions.

**5. Dataset novelty**

The claim that GloVe-100 is an independent evaluation dataset is bounded to the dataset lists examined in the two papers. It is not a claim that the dataset has never been used in previous clustering research.

## Repository structure

```text
keo/
├── configs/              # Dataset and experiment configurations
├── src/
│   └── keo/
│       ├── algos/        # k-means, e-means, o-means
│       │   └── adapters/ # Official implementation adapters
│       ├── bench/        # Experiment runner and result storage
│       ├── metrics/      # Evaluation metrics and operation counting
│       ├── analysis/     # Figure generation and analysis
│       └── data/         # Download, verification, preparation
├── tests/                # Unit and phase-gate tests
├── results/              # Experiment outputs
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
└── LICENSE
```

## References

### Clustering algorithms

- **e-means:** Bessa et al. (2026). *e-means: Outperforming k-means (in one-tenth the epochs).* [Paper](https://arxiv.org/abs/2611.09501) · [Project page](https://e-means.github.io)
- **o-means:** Bajaj, Nupur, Garg (2026). *o-means: Index the centroids, don't scan them.* Project page, November 2026. Paper not yet released.
- **Flash-KMeans:** GPU k-means baseline. [arXiv:2603.09229](https://arxiv.org/abs/2603.09229)
- **Faiss:** Douze et al. (2024). [arXiv:2401.08281](https://arxiv.org/abs/2401.08281)

### Related research

- **VQ-VAE:** van den Oord et al. (2017). [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- **ANN-Benchmarks:** Aumüller et al. (2018). [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)
- **GloVe:** Pennington et al. [Project page](https://nlp.stanford.edu/projects/glove/)
- **VIBE:** [arXiv:2505.17810](https://arxiv.org/abs/2505.17810)

### Classical foundations

- Lloyd (1982). *Least squares quantization in PCM.*
- Arthur and Vassilvitskii (2007). *k-means++: The advantages of careful seeding.*
- Sculley (2010). *Web-scale k-means clustering.*

## Contributing

Contributions are welcome, and you do not need to write code to help. Running the benchmark on a GPU, suggesting a dataset, or reviewing the o-means reconstruction are all useful. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up, what a good pull request looks like, and the rules that keep the benchmark trustworthy. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). To report a security problem, see [SECURITY.md](SECURITY.md).

If you use KEO in your work, please cite it using [CITATION.cff](CITATION.cff) or the DOI badge above.

## License

KEO is released under the [MIT License](https://github.com/rajveer43/keo/blob/main/LICENSE).

The GloVe vectors are not redistributed with the package. They are downloaded from ANN-Benchmarks and remain subject to their respective terms.

---

<p align="center">
  <b>KEO — Reproducible clustering research.</b>
  <br>
  Built for transparent, controlled, and reproducible evaluation.
</p>