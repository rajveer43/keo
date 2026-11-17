# Contributing to KEO

Thank you for your interest in KEO. This is a benchmark, so the most valuable contributions are the ones that make its results more trustworthy: more hardware, more datasets, more careful checks of the algorithms.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

| You want to... | Do this |
|---|---|
| Report something that looks wrong | Open a [bug report](https://github.com/rajveer43/keo/issues/new?template=bug_report.yml) |
| Run the benchmark on a GPU or other hardware | Open a [benchmark result](https://github.com/rajveer43/keo/issues/new?template=benchmark_result.yml) issue |
| Suggest a dataset, algorithm, metric or figure | Open a [feature request](https://github.com/rajveer43/keo/issues/new?template=feature_request.yml) |
| Check the o-means reconstruction | Read the assumptions list (see below) and open an issue or a pull request |
| Fix a typo or improve the docs | Open a pull request directly |
| Ask a question | Use [Discussions](https://github.com/rajveer43/keo/discussions), or an issue if Discussions is off |

For anything larger than a small fix, please open an issue first so we can agree on the approach before you spend time on it.

## Development setup

You need Python 3.8 or newer (3.12 recommended) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<your-username>/keo.git
cd keo
uv sync
uv run pytest
```

The tests run on a CPU and do not need the dataset. Tests that need the downloaded GloVe file are marked `data` and skip themselves when it is absent. To run them, first run `uv run keo download glove100`.

```bash
uv run pytest -m "not slow"     # skip long-running tests
uv run pytest -m data           # only tests that need the dataset
uv run keo smoke                # end-to-end check, about 10 seconds
```

## Making a change

1. Fork the repository and create a branch from `main`, for example `fix-ivf-rounding` or `add-sift-dataset`.
2. Make the change. Keep it focused: one topic per pull request.
3. Add or update tests. A bug fix should come with a test that fails without the fix.
4. Run `uv run pytest` and make sure it passes.
5. Update the README or the relevant docs if behaviour changes.
6. Open a pull request using the template. Describe what changed and why.

### Commit messages

Use short, imperative subject lines, such as `Fix dead-cluster count for empty batches`. Explain the reason in the body when it is not obvious.

### Code style

- Follow the style of the surrounding code. There is no formatter enforced yet; keep lines readable and avoid unrelated reformatting in your diff.
- Use type hints on public functions.
- Prefer clear names and short functions over comments that restate the code. Comment the *why*, especially where a choice follows a paper or the official implementation.
- Do not add new required dependencies without discussing it first. The package is installed by researchers, and heavy dependencies cost them time.

## Rules specific to a benchmark

These matter more than style, because they protect the validity of the results.

**Reproducibility.** Every random choice must come from an explicit seed. A result must be reproducible from its config alone. Results are stored under a content hash of their config; do not change what goes into that hash without a good reason and a note in the pull request.

**Reference vs official implementations.** Our own PyTorch implementations and the authors' official code are kept separate and are never mixed inside one experiment. Wall-clock time is compared only within the same implementation. Do not add a comparison that breaks this.

**Document assumptions.** Where a paper is silent or ambiguous (this applies to o-means especially, whose paper is unreleased), do not guess silently. Record the assumption with the reason, so a reader can challenge it. In a source checkout these live in `docs/ASSUMPTIONS.md`; if you do not have that file, describe the assumption in your pull request and we will add it.

**No hand-typed results.** Numbers in figures and the README must be computed from stored results. Do not paste a number into a figure or caption.

**Honest labelling.** Results from small or smoke runs must say so. A claim about speed on a GPU needs a GPU measurement; a claim about operation counts does not.

## Adding a dataset

1. Add `configs/datasets/<name>.yaml`, following `glove100.yaml`: source URL, file format, keys, dimensions, sizes, normalisation, split.
2. Pin a SHA-256 checksum of the file. If the publisher does not provide one, say so in the config comment and record the checksum from your first download.
3. State whether either source paper evaluated this dataset. The benchmark's central claim depends on using data the methods were not tuned on.
4. Add a test that loads the config and checks its shape fields.

## Adding an algorithm

1. Put the implementation in `src/keo/algos/`, following the interface in `base.py`.
2. Count distance computations the same way the existing algorithms do, so costs stay comparable.
3. Add tests for correctness (for example, that it reduces the objective and handles empty clusters) and for determinism under a fixed seed.
4. If an official implementation exists, add an adapter under `src/keo/algos/adapters/` that runs it in a separate worker process, and record which implementation produced each result.

## Submitting benchmark results

You do not need to write code to help. If you run an experiment on hardware we do not have:

1. Install KEO (`pip install keo`) and run the same experiment config you are reporting.
2. Open a [benchmark result](https://github.com/rajveer43/keo/issues/new?template=benchmark_result.yml) issue with the hardware, library versions, config, and the output of `keo status <name>`.
3. Do not edit the numbers. Attach the results file instead, so we can reproduce the figures from it.

## Reviews

A maintainer will review your pull request when they can, usually within a week or two. Reviews focus on correctness and on whether the change preserves the benchmark's guarantees. Please be patient and keep discussion kind and specific.

## Releases

Maintainers publish releases. The process is in [RELEASING.md](RELEASING.md). Contributors should not change version numbers.

## Licence

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE), the same as the rest of the project.
