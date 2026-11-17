# Changelog

All notable changes to KEO are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.4]

### Added

- Contributor documentation: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, issue and pull request templates.
- Package metadata: PyPI keywords, classifiers and project URLs.

### Fixed

- `keo` CLI options declared with `X | None` crashed on Python 3.9 and earlier; they now use `Optional[...]`, and the CLI module no longer uses string annotations so Typer keeps its `Annotated` option metadata on 3.8/3.9.
- `test_parity_pairs_same_seed_and_budget` read its CSV with pandas' fast float parser, which is not round-trip exact and made the test fail by roundoff on one Python/pandas combination; it now reads with `float_precision="round_trip"` (tolerance unchanged).

### Changed

- Python support widened from 3.10+ to 3.8 through 3.14. The tests run on every one of these versions in CI.
- Dependency lower bounds are now split by Python version: Python 3.9+ keeps the previous bounds, and Python 3.8 uses older releases (numpy 1.24, scipy 1.10, scikit-learn 1.3, pandas 2.0, matplotlib 3.7, faiss-cpu 1.7.4). `typing_extensions` is required on Python 3.8 only.
- The official faiss E-means adapter needs faiss-cpu >= 1.16, which requires Python >= 3.10, so on Python 3.8 and 3.9 it reports itself as unavailable.
- No algorithm behaviour changed. On Python 3.8 results may differ slightly in the last digits because of the older library versions.

## [0.1.3]

- Package metadata and Zenodo archive fixes.

## [0.1.2]

- Added `CITATION.cff` and the DOI badge.

## [0.1.1]

- Added badges and the tests workflow.

## [0.1.0]

- First release: k-means, e-means and o-means reference implementations, official-code adapters, resumable runner, six figures, and the GloVe-100 pipeline.
