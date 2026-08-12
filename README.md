# depiction

This package provides functionality to process and visualize mass-spectrometry imaging data.
Currently, it requires your data to be available in the `imzML` format.

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with three
packages:

- `depiction` (in `src/`): image and spectrum processing, plus `depiction_targeted_preproc`, a
  pipeline that turns an acquisition into calibrated spectra, ion images and QC plots.
- `depiction_io` (in `pkgs/depiction_io/`): reading and writing MSI data. Everything else talks to
  the protocols in `depiction_io.types` rather than to a file format. **If reading and writing MSI
  files is all you need, depend on this package rather than on `depiction`** — see
  [`pkgs/depiction_io/README.md`](https://github.com/fgcz/depiction/blob/dev/pkgs/depiction_io/README.md).
- `snakemake_invoke` (in `pkgs/snakemake_invoke/`): a small wrapper for invoking the pipeline's
  Snakemake workflow from Python — see
  [`pkgs/snakemake_invoke/README.md`](https://github.com/fgcz/depiction/blob/dev/pkgs/snakemake_invoke/README.md).

## Scope

Two things are tested and exercised end to end: **`depiction_io`**, which has its own test suite, a
protocol conformance test per backend and a differential corpus; and **the targeted preprocessing
pipeline**, which is covered by [system tests](https://github.com/fgcz/depiction/blob/dev/system_tests/README.md) against two real
acquisitions and whose output is pinned against the pre-migration implementation in
[`docs/refactoring/baseline-diff.md`](https://github.com/fgcz/depiction/blob/dev/docs/refactoring/baseline-diff.md).

The following are in the tree but do not work:

- **The clustering _tools_** — `depiction.tools.clustering` and `depiction.clustering`. They
  import, but every path through them raises `TypeError` as soon as it is called: three call sites
  construct a `MultiChannelImage` without the required `is_foreground`. Repairing those would
  expose `clustering/maxmin_sampling.py`, which ignores its `metric` argument after the first pick
  — reviewing that sampling geometry is a research question rather than a maintenance task, which
  is why the obvious failure was left in place. This does *not* include the pipeline's own
  `workflow/proc/cluster_kmeans.py` and `cluster_hdbscan.py`, which are separate code, work, and
  are covered by tests.
- **`depiction_cluster_sandbox`** — needs a data directory that no longer exists, and `umap`, which
  is declared in no extra.
- A handful of Snakemake rules that no artifact can reach, and the one-off scripts under
  `workflow/exp/`. Both are listed in
  [`src/depiction_targeted_preproc/README.md`](https://github.com/fgcz/depiction/blob/dev/src/depiction_targeted_preproc/README.md).

## Setup dev environment

Python 3.13 is required.

### Install with `uv`

The application [uv](https://github.com/astral-sh/uv) provides both very fast installation of all required dependencies, as well as functionality to install a particular version of Python for you.

If you do not have `uv` installed yet, please consult their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

To create the virtual environment and install every workspace package in editable mode (i.e. changes to code are immediately available in the environment), run:

```bash
uv sync --extra dev
```

This creates the virtual environment in the `.venv` directory.

Add `--extra findmf` if you need the FindMF peak picker (the `dev_presets` and
`dev_no_calibration` pipeline presets use it). It is optional because `findmfpy` is a C++
extension that ships wheels only for CPython 3.13 on macOS arm64 and x86-64 Linux; anywhere
else, installing it means compiling it.
To activate the environment in your shell, you need to `source` the correct activation script from `.venv/bin`, e.g. `.venv/bin/activate` for bash.

If you use an IDE you may want to point the IDE to the Python interpreter at `.venv/bin/python`.

### Set up `pre-commit`

To check and format the code automatically, you can use `pre-commit`.
In general, you can use the latest version.

```bash
pipx install pre-commit
pre-commit install
```

Now, the checks will be run automatically before each commit.
The first time you might have some delay because the hooks are installed.

### Test with `nox`

To run the tests the same way as in the CI, you can use `nox`.
In general, you can use the latest version.

```bash
pipx install nox
```

Then you can run the checks with

```bash
nox
```

or one session at a time:

```bash
nox -s tests_depiction
nox -s tests_depiction_io
```

`nox -l` lists all of them. `tests_depiction_io` deliberately installs only `depiction_io`, so an
accidental dependency on `depiction` fails there rather than being masked by the parent
environment. `docs`, `licensecheck`, `system_tests` and `tests_structure` are kept off the default
run — each session's docstring says why — and CI runs all but `tests_structure` in their own jobs.

However, you can also run the tests with `pytest` or from your IDE if you are in the virtual environment.

## Running the pipeline

The targeted preprocessing pipeline takes a **chunk directory** — one acquisition plus the
configuration for it — and builds whatever that configuration asks for:

```bash
python -m depiction_targeted_preproc.app_interface.process_chunk /path/to/work/my_sample
```

[`src/depiction_targeted_preproc/README.md`](https://github.com/fgcz/depiction/blob/dev/src/depiction_targeted_preproc/README.md) describes
the directory layout, `params.yml`, the available artifacts and how to add one.
[`system_tests/`](https://github.com/fgcz/depiction/blob/dev/system_tests/README.md) is a worked example that runs on a public 59 MB
acquisition a fresh clone can fetch.

## Documentation

- [`docs/`](https://github.com/fgcz/depiction/tree/dev/docs) — API documentation, built with `nox -s docs`. The dimension and geometry
  conventions the image classes follow, including which way the y axis points, are described in
  [`docs/modules/image/multi_channel_image.md`](https://github.com/fgcz/depiction/blob/dev/docs/modules/image/multi_channel_image.md).
- [`src/depiction_targeted_preproc/README.md`](https://github.com/fgcz/depiction/blob/dev/src/depiction_targeted_preproc/README.md) — the
  pipeline: how to run it, what it produces, and which rules are known to be dead.
- [`system_tests/README.md`](https://github.com/fgcz/depiction/blob/dev/system_tests/README.md) — the end-to-end recipe and its fixtures.
- [`docs/refactoring/ROADMAP.md`](https://github.com/fgcz/depiction/blob/dev/docs/refactoring/ROADMAP.md) — what was migrated to `imzy`, what
  was deliberately left undone, and why.
