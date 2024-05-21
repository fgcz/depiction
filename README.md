## Status
A lot of the files and folders are in a early development state or may require some serious refactoring.
For now, I give a summary of the modules which are somewhat stable already:

| Module                      | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------| 
| `ionplotter.calibration`    | Calibration specific functionality.                                         |
| `ionplotter.image`          | Image specific functionality.                                               |
| `ionplotter.parallel_ops`   | Basic building blocks for parallelization of operations accross *ReadFiles. |
| `ionplotter.persistence`    | Persistence specific code, including reader and writer for Imzml files.     |
| `ionplotter.peak_filtering` | Peak filtering specific code.                                               |
| `ionplotter.peak_picking`   | Peak picking specific code.                                                 |
| `ionplotter.visualize`      | Visualization specific code.                                                |


## Setup dev environment
Install the `pyproject.toml` in editable mode, tested with [uv](https://github.com/astral-sh/uv):

```bash
uv venv # <- if not using anaconda
uv pip install -e ".[dev,testing]"
```
