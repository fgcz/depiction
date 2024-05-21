## Status
A lot of the files and folders are in a early development state or may require some serious refactoring.
For now, I give a summary of the modules which are somewhat stable already:

## Setup dev environment
Install the `pyproject.toml` in editable mode, tested with [uv](https://github.com/astral-sh/uv):

```bash
uv venv # <- if not using anaconda
uv pip install -e ".[dev,testing]"
```
