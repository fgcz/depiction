# depiction

This package provides functionality to process and visualize mass-spectrometry imaging data.
Currently, it requires your data to be available in the `imzML` format.
The full pipeline is also in the process of being developed.

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two
packages:

- `depiction` (in `src/`): implements the whole functionality to process the data. It also contains
  `depiction_targeted_preproc`, a pipeline that based on some configuration file creates outputs like
  a qc report and .ome.tiff files.
- `depiction_io` (in `pkgs/depiction_io/`): reading and writing MSI data. Everything else talks to
  the protocols in `depiction_io.types` rather than to a file format, so the storage backend can be
  changed without touching the callers. See
  [`pkgs/depiction_io/README.md`](pkgs/depiction_io/README.md).

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

However, you can also run the tests with `pytest` or from your IDE if you are in the virtual environment.

## Geometry Conventions

TODO these are not used consistently everywhere yet

### Dimension names

- (2D) Points: (x, y)
- (2D) Images: (y, x, c)
- Sparse images: (i, c)
- Coordinates: (i, d) and each row corresponds to point ordering (i.e. (x, y))

TODO: y-axis direction, xarray conventions (dims, coords, etc.)
