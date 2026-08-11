# Stale `build/lib` trees can resurrect deleted modules into an in-tree install

Severity: **low** | Status: **partly fixed** | Found: 2026-08-10
Files: `build/`, `pkgs/depiction_io/build/`, `pkgs/snakemake_invoke/build/`
(untracked); note at `pyproject.toml:118-121`

## Symptom

`pip install .` run from a working copy that still has an old `build/lib` can ship modules
that were deleted from `src/` months ago. In this working copy those trees still contain the
entire hand-rolled imzML parser removed by the imzy migration:

```
pkgs/depiction_io/build/lib/depiction_io/imzml/imzml_read_file.py
pkgs/depiction_io/build/lib/depiction_io/imzml/imzml_reader.py
pkgs/depiction_io/build/lib/depiction_io/imzml/compression.py
pkgs/depiction_io/build/lib/depiction_io/imzml/parser/parse_spectra.py
pkgs/depiction_io/build/lib/depiction_io/imzml/parser/cv_params.py
pkgs/depiction_io/build/lib/depiction_io/imzy_backend/compression_guard.py
```

None of these exist under `git ls-files`.

## Why it happens

setuptools copies `src/` into `build/lib` and then packages `build/lib`, without pruning files
that no longer have a source. This is exactly the failure `pyproject.toml:118-121` documents
from two years ago, when a stale `build/lib` shipped an obsolete `snakemake_invoke` inside the
`depiction` wheel and shadowed the real package. The explicit
`[tool.setuptools.packages.find]` added then prevents the *package discovery* half of that
problem, but not this one.

## Scope

`build/` is gitignored and untracked, so **CI and any fresh clone are unaffected**, and
`uv build` is clean. The exposure is confined to installs made from this working copy — i.e.
to you, and to anyone you hand a tarball of the directory.

## Fix

**Done:** the assurance at `pyproject.toml:118-121` has been softened. It read as though the
packages-find setting closed this off, and it did not; the comment now says which half it
covers, and names the command below as the fix.

**Still open**, because it is a local filesystem action on untracked directories and cannot
live in a commit:

```bash
rm -rf build/ pkgs/*/build/
```
