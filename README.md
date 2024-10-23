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

### Using `uv` (recommended)

Install the `pyproject.toml` in editable mode, tested with
[uv](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -e ".[dev]"
```

### Using `conda`

```bash
conda create -n exp-2024-05-depiction python=3.11
conda activate exp-2024-05-depiction
pip install -e ".[dev]"
```

## Geometry Conventions

TODO these are not used consistently everywhere yet

### Dimension names

- (2D) Points: (x, y)
- (2D) Images: (y, x, c)
- Sparse images: (i, c)
- Coordinates: (i, d) and each row corresponds to point ordering (i.e. (x, y))

TODO: y-axis direction, xarray conventions (dims, coords, etc.)
