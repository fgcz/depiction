# Refactoring Plan: Convert to UV Workspace with depiction_io Package

## Context

The depiction repository currently contains mass spectrometry I/O operations mixed with higher-level image processing and analysis code. This creates unnecessary dependencies for users who only need the I/O functionality. The goal is to:

1. Extract `src/depiction/persistence/` → `pkgs/depiction_io/` as a standalone package
2. Convert the repository to a UV workspace supporting multiple packages
3. Make `pyimzml` and related I/O dependencies internal to `depiction_io` only
4. Enable independent versioning and distribution of I/O functionality

**Impact:** 124 import statements across 94 files will need updating.

## Key Architectural Decision

**Handling the Circular Dependency:**

There's a circular import between:
- `depiction.image.MultiChannelImage` imports `depiction.persistence.image.hdf5_image_format.Hdf5ImageFormat`
- `Hdf5ImageFormat` imports `MultiChannelImage`

**Solution:** Keep `Hdf5ImageFormat` in `depiction.image` (move it from `persistence/image/` to `image/`). This is architecturally sound because:
- `Hdf5ImageFormat` is a convenience wrapper for `MultiChannelImage` persistence (tightly coupled)
- It's not mass-spec specific (works for any multi-channel image)
- Avoids circular package dependency between `depiction` and `depiction_io`

All other persistence modules (imzML, OME-TIFF, RAM) will move to `depiction_io`.

## Critical Files

**Configuration:**
- `pyproject.toml` (root) - Add workspace config, remove pyimzml/bioio deps
- `pkgs/depiction_io/pyproject.toml` (new) - Package config with I/O dependencies

**Source files to move (24 files):**
- `src/depiction/persistence/imzml/` → `pkgs/depiction_io/src/depiction_io/imzml/` (12 files)
- `src/depiction/persistence/ram/` → `pkgs/depiction_io/src/depiction_io/ram/` (4 files)
- `src/depiction/persistence/image/` → `pkgs/depiction_io/src/depiction_io/image/` (3 files, excluding hdf5)
- `src/depiction/persistence/{types.py,file_checksums.py,imzml_zip.py}` → `pkgs/depiction_io/src/depiction_io/`

**File to relocate (not to depiction_io):**
- `src/depiction/persistence/image/hdf5_image_format.py` → `src/depiction/image/hdf5_image_format.py`

**Test files to move (13 files):**
- `tests/unit/persistence/` → `pkgs/depiction_io/tests/unit/`

**Import updates needed in:**
- `src/depiction/` - 47 files
- `tests/` - 38 files
- `src/depiction_targeted_preproc/` - 12 files

## Implementation Steps

### Phase 1: Workspace Structure Setup

**1.1 Create directory structure:**
```bash
mkdir -p pkgs/depiction_io/src/depiction_io/{imzml,ram,image}
mkdir -p pkgs/depiction_io/tests/unit/{imzml,ram,image}
```

**1.2 Create workspace root config:**

Update `pyproject.toml` to add at the top:
```toml
[tool.uv.workspace]
members = [".", "pkgs/*"]
```

And remove these dependencies (moving to depiction_io):
- `pyimzml>=1.5.3`
- `bioio`
- `bioio-ome-tiff>=1.4.0`
- `bioio-ome-zarr`
- `tifffile`
- `spatialdata>=0.7.2`

Add instead:
```toml
dependencies = [
    "depiction_io",  # workspace dependency
    # ... keep all other dependencies
]
```

**1.3 Create depiction_io package config:**

Create `pkgs/depiction_io/pyproject.toml`:
```toml
[project]
name = "depiction_io"
version = "0.1.0"
description = "I/O operations for mass spectrometry imaging data"
authors = [{ name = "Leonardo Schwarz", email = "leonardo.schwarz@fgcz.ethz.ch" }]
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">= 3.13"

dependencies = [
    "numpy>=2.0.0",
    "xarray",
    "pydantic>=2.6.0",
    "pyimzml>=1.5.3",
    "bioio",
    "bioio-ome-tiff>=1.4.0",
    "tifffile",
    "loguru",
    "spatialdata>=0.7.2",
    "h5netcdf",
    "tqdm>=4.66.1",
    "cyclopts",
    "depiction",  # for MultiChannelImage import in ome_tiff.py
]

[project.optional-dependencies]
testing = [
    "pytest",
    "pytest-mock",
    "hypothesis>=6.99.11",
]

dev = [
    "depiction_io[testing]",
    "ruff>=0.3.5",
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 120
indent-width = 4
target-version = "py313"

[tool.ruff.lint]
select = ["ANN", "BLE", "D103", "E", "F", "PLW", "PTH", "SIM", "UP", "TCH"]
ignore = ["ANN101", "ANN102", "TCH002"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**1.4 Create README:**

Create `pkgs/depiction_io/README.md`:
```markdown
# depiction_io

I/O operations for mass spectrometry imaging data formats.

## Features

- ImzML reading and writing (pyimzml wrapper)
- OME-TIFF image I/O
- RAM-based format for in-memory processing
- File checksum utilities

## Installation

```bash
pip install depiction_io
```
```

### Phase 2: Move Source Files

**2.1 Move persistence module to depiction_io:**
```bash
# Move directories
mv src/depiction/persistence/imzml pkgs/depiction_io/src/depiction_io/imzml
mv src/depiction/persistence/ram pkgs/depiction_io/src/depiction_io/ram

# Move image directory (will need to remove hdf5_image_format.py separately)
mv src/depiction/persistence/image pkgs/depiction_io/src/depiction_io/image

# Move root files
mv src/depiction/persistence/types.py pkgs/depiction_io/src/depiction_io/
mv src/depiction/persistence/file_checksums.py pkgs/depiction_io/src/depiction_io/
mv src/depiction/persistence/imzml_zip.py pkgs/depiction_io/src/depiction_io/
```

**2.2 Move hdf5_image_format.py to depiction.image:**
```bash
mv pkgs/depiction_io/src/depiction_io/image/hdf5_image_format.py src/depiction/image/
```

**2.3 Create depiction_io __init__.py:**

Create `pkgs/depiction_io/src/depiction_io/__init__.py`:
```python
"""I/O operations for mass spectrometry imaging data."""

from depiction_io.imzml.imzml_mode_enum import ImzmlModeEnum
from depiction_io.imzml.imzml_read_file import ImzmlReadFile
from depiction_io.imzml.imzml_reader import ImzmlReader
from depiction_io.imzml.imzml_write_file import ImzmlWriteFile
from depiction_io.imzml.imzml_writer import ImzmlWriter
from depiction_io.ram.ram_read_file import RamReadFile
from depiction_io.ram.ram_reader import RamReader

__all__ = [
    "ImzmlModeEnum",
    "ImzmlReadFile",
    "ImzmlReader",
    "ImzmlWriteFile",
    "ImzmlWriter",
    "RamReadFile",
    "RamReader",
]
```

**2.4 Update persistence __init__.py for backward compatibility:**

Edit `src/depiction/persistence/__init__.py`:
```python
"""Re-exports from depiction_io for backward compatibility.

Deprecated: Import from depiction_io directly instead.
"""

from depiction_io import (
    ImzmlModeEnum,
    ImzmlReadFile,
    ImzmlReader,
    ImzmlWriteFile,
    ImzmlWriter,
    RamReadFile,
    RamReader,
)

__all__ = [
    "ImzmlModeEnum",
    "ImzmlReadFile",
    "ImzmlReader",
    "ImzmlWriteFile",
    "ImzmlWriter",
    "RamReadFile",
    "RamReader",
]
```

### Phase 3: Update Import Statements

**3.1 Update imports in depiction_io package (internal):**

In all moved files under `pkgs/depiction_io/src/depiction_io/`, replace:
- `from depiction.persistence.` → `from depiction_io.`
- `import depiction.persistence.` → `import depiction_io.`

Keep imports from `depiction.image` as-is (external dependency).

**3.2 Update imports in main depiction package:**

In all files under `src/depiction/` (except `persistence/__init__.py`), replace:
- `from depiction.persistence import` → `from depiction_io import`
- `from depiction.persistence.` → `from depiction_io.`

Special case in `src/depiction/image/multi_channel_image.py`:
```python
# OLD (line 13)
from depiction.persistence.image.hdf5_image_format import Hdf5ImageFormat

# NEW
from depiction.image.hdf5_image_format import Hdf5ImageFormat
```

**3.3 Update imports in tests:**

In all files under `tests/`, replace:
- `from depiction.persistence import` → `from depiction_io import`
- `from depiction.persistence.` → `from depiction_io.`

**3.4 Update imports in depiction_targeted_preproc:**

In all files under `src/depiction_targeted_preproc/`, replace:
- `from depiction.persistence import` → `from depiction_io import`
- `from depiction.persistence.` → `from depiction_io.`

**Automation approach:** Use find/sed for bulk replacement:
```bash
find src/depiction tests src/depiction_targeted_preproc -name "*.py" -type f -exec sed -i '' \
  -e 's/from depiction\.persistence import/from depiction_io import/g' \
  -e 's/from depiction\.persistence\./from depiction_io./g' \
  {} +

# Fix the special case for hdf5_image_format
sed -i '' 's/from depiction_io\.image\.hdf5_image_format/from depiction.image.hdf5_image_format/g' \
  src/depiction/image/multi_channel_image.py
```

### Phase 4: Move Tests

**4.1 Move test files:**
```bash
mv tests/unit/persistence/imzml pkgs/depiction_io/tests/unit/imzml
mv tests/unit/persistence/ram pkgs/depiction_io/tests/unit/ram
mv tests/unit/persistence/image pkgs/depiction_io/tests/unit/image
mv tests/unit/persistence/test_file_checksums.py pkgs/depiction_io/tests/unit/
```

**4.2 Update test imports:**

In moved test files under `pkgs/depiction_io/tests/`, replace:
- `from depiction.persistence import` → `from depiction_io import`
- `from depiction.persistence.` → `from depiction_io.`

Keep imports from `depiction.image` (external dependency).

**4.3 Create test config:**

Create `pkgs/depiction_io/tests/conftest.py`:
```python
import warnings
import pytest


@pytest.fixture()
def treat_warnings_as_error():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
```

### Phase 5: Cleanup

**5.1 Remove empty persistence subdirectories:**
```bash
# After moving files, only __init__.py should remain
rm -rf src/depiction/persistence/imzml
rm -rf src/depiction/persistence/ram
rm -rf src/depiction/persistence/image
rm -rf tests/unit/persistence
```

**5.2 Add type marker:**
```bash
touch pkgs/depiction_io/src/depiction_io/py.typed
```

Update `pkgs/depiction_io/pyproject.toml`:
```toml
[tool.setuptools.package-data]
depiction_io = ["py.typed"]
```

**5.3 Update main pyproject.toml setuptools config:**

Ensure package discovery is explicit:
```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["depiction*"]
```

## Verification Steps

### Step 1: Install workspace
```bash
cd /Users/leo/code/depiction/feats/separate-depiction-io
uv sync
```

Expected: No errors, both packages installed in editable mode.

### Step 2: Test depiction_io package
```bash
uv run pytest pkgs/depiction_io/tests/ -v
```

Expected: All 13+ test files pass.

### Step 3: Test main depiction package
```bash
uv run pytest tests/ -v
```

Expected: All remaining tests pass.

### Step 4: Verify imports work
```bash
uv run python -c "
from depiction_io import ImzmlReader, ImzmlWriter, ImzmlModeEnum, RamReadFile
from depiction.persistence import ImzmlReader as BackwardCompat  # backward compat
from depiction.image.hdf5_image_format import Hdf5ImageFormat
from depiction.image import MultiChannelImage
print('✓ All import patterns work')
"
```

Expected: No import errors.

### Step 5: Verify pyimzml is only in depiction_io
```bash
uv pip show depiction | grep -i pyimzml  # Should return nothing
uv pip show depiction_io | grep -i pyimzml  # Should show pyimzml
```

Expected: pyimzml only appears in depiction_io dependencies.

### Step 6: Run full test suite via nox
```bash
nox
```

Expected: All sessions pass (lint, tests, licensecheck).

### Step 7: Test CLI tools
```bash
depiction-tools --help
# Try a command that uses persistence
depiction-tools imzml --help
```

Expected: CLI works without errors.

## Rollback Strategy

If issues arise:
1. Revert all commits made during refactoring
2. Workspace structure can be removed by deleting `pkgs/` directory and `[tool.uv.workspace]` section
3. Git reflog can recover moved files if needed

Keep backups before starting:
```bash
git checkout -b backup-before-workspace-refactor
```

## Post-Implementation Tasks

1. Update CI/CD workflow to test both packages
2. Create migration guide (MIGRATION.md) for users
3. Update README with workspace structure explanation
4. Consider deprecation warnings for `depiction.persistence` imports (future)
5. Version bump: depiction 0.1.0 → 0.2.0, depiction_io starts at 0.1.0

## Estimated Effort

- Setup and file moves: 2-3 hours
- Import updates: 3-4 hours
- Testing and verification: 2-3 hours
- Documentation: 1-2 hours

**Total: 8-12 hours**
