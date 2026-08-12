# depiction

This package provides functionality to process and visualize mass-spectrometry imaging data.
Currently, it requires your data to be available in the `imzML` format.
The full pipeline is also in the process of being developed.

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with three
packages:

- `depiction` (in `src/`): implements the whole functionality to process the data. It also contains
  `depiction_targeted_preproc`, a pipeline that based on some configuration file creates outputs like
  a qc report and .ome.tiff files.
- `depiction_io` (in `pkgs/depiction_io/`): reading and writing MSI data. Everything else talks to
  the protocols in `depiction_io.types` rather than to a file format, so the storage backend can be
  changed without touching the callers. **If reading and writing MSI files is all you need, depend
  on this package rather than on `depiction`** — see
  [`pkgs/depiction_io/README.md`](pkgs/depiction_io/README.md).
- `snakemake_invoke` (in `pkgs/snakemake_invoke/`): a small wrapper for invoking the pipeline's
  Snakemake workflow from Python, vendored from its own repository so that this one has no git
  dependencies. See [`pkgs/snakemake_invoke/README.md`](pkgs/snakemake_invoke/README.md).

This project is in an early state of development. If you are interested, it's best to reach out to us.

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

or more specifically, one package's tests at a time:

```bash
nox -s tests_depiction
nox -s tests_depiction_io
```

`tests_depiction_io` deliberately installs only `depiction_io`, so an accidental dependency on
`depiction` fails there rather than being masked by the parent environment.

`nox -s docs` builds the Sphinx documentation with warnings treated as errors, so a broken autodoc
reference fails the build rather than silently dropping a page. It is not part of the default run:
intersphinx fetches remote inventories, so an upstream site moving one would fail a bare `nox` on a
commit that changed nothing. `nox -s licensecheck` is non-default for the same reason. CI runs both
in their own job. The slow end-to-end `system_tests` session is not a default either; see
[system_tests/README.md](system_tests/README.md).

However, you can also run the tests with `pytest` or from your IDE if you are in the virtual environment.

## Geometry Conventions

TODO these are not used consistently everywhere yet

### Dimension names

- (2D) Points: (x, y)
- (2D) Images: (y, x, c)
- Sparse images: (i, c)
- Coordinates: (i, d) and each row corresponds to point ordering (i.e. (x, y))

TODO: y-axis direction, xarray conventions (dims, coords, etc.)
