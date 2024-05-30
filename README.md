## Status

A lot of the files and folders are in a early development state or may require some serious refactoring.
For now, I give a summary of the modules which are somewhat stable already:

| Module                              | Description                                                                |
|-------------------------------------|----------------------------------------------------------------------------|
| `depiction.spectrum`                | Collection of spectra processing functionality.                            |
| `depiction.spectrum.baseline`       | Estimate spectrum baseline curve.                                          |
| `depiction.spectrum.peak_picking`   |                                                                            |
| `depiction.spectrum.peak_filtering` |                                                                            |
| `depiction.calibration`             | Calibration specific functionality.                                        |
| `depiction.image`                   | Image specific functionality.                                              |
| `depiction.parallel_ops`            | Basic building blocks for parallelization of operations across *ReadFiles. |
| `depiction.persistence`             | Persistence specific code, including reader and writer for Imzml files.    |
| `depiction.peak_filtering`          | Peak filtering specific code.                                              |
| `depiction.peak_picking`            | Peak picking specific code.                                                |
| `depiction.visualize`               | Visualization specific code.                                               |

## Setup dev environment

Install the `pyproject.toml` in editable mode, tested with [uv](https://github.com/astral-sh/uv):

```bash
uv venv # <- if not using anaconda
uv pip install -e ".[dev,testing]"
```

TODO conda install:

```bash
conda create -n exp-2024-05-depiction python=3.12 
conda activate exp-2024-05-depiction
pip install --system -e ".[dev,testing]" # rerun if this fails
```

## Geometry Conventions

TODO these are not used consistently everywhere yet

### Dimension names

- (2D) Points: (x, y)
- (2D) Images: (y, x, c)
- Sparse images: (i, c)
- Coordinates: (i, d) and each row corresponds to point ordering (i.e. (x, y))

TODO: y-axis direction, xarray conventions (dims, coords, etc.)
