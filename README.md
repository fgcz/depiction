# depiction

This package provides functionality to process and visualize mass-spectrometry imaging data.
Currently, it requires your data to be available in the `imzML` format.
The full pipeline is also in the process of being developed.

The project is structured in two general parts:

- `depiction`: implements the whole functionality to process the data
- `depiction_targeted_preproc`: implements a pipeline that based on some configuration file creates outputs like qc report and .ome.tiff files

This project is in an early state of development. If you are interested, it's best to reach out to us.

## Setup dev environment

Currently, Python 3.12 is required, 3.13 is not compatible yet (missing wheels and e.g. llvmlite).

### Install `uv`

The application [uv](https://github.com/astral-sh/uv) provides both very fast installation of all required dependencies, as well as functionality to install a particular version of Python for you.

If you do not have `uv` installed yet, please consult their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

To create a virtual environment using `uv` and install depiction in editable mode (i.e. changes to code are immediately available in the environment), run the following commands:

```bash
uv venv -p 3.12
uv pip install -e ".[dev]"
```

This creates the virtual environment in the `.venv` directory.
To activate the environment in your shell, you need to `source` the correct activation script from `.venv/bin`, e.g. `.venv/bin/activate` for bash.

### Install `pre-commit`

To check and format the code automatically, you can use `pre-commit`.
In general, you can use the latest version.

```bash
pipx install pre-commit
pre-commit install
```

Now, the checks will be run automatically before each commit.
The first time you might have some delay because the hooks are installed.

### Install `nox`

To run the tests the same way as in the CI, you can use `nox`.
In general, you can use the latest version.

```bash
pipx install nox
```

Then you can run the checks with

```bash
nox
```

or more specifically:

```bash
nox -s tests
```

## Geometry Conventions

TODO these are not used consistently everywhere yet

### Dimension names

- (2D) Points: (x, y)
- (2D) Images: (y, x, c)
- Sparse images: (i, c)
- Coordinates: (i, d) and each row corresponds to point ordering (i.e. (x, y))

TODO: y-axis direction, xarray conventions (dims, coords, etc.)
