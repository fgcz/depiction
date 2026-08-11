# `ImzmlZip.extract` extracts into directories and returns a path that does not exist

Severity: **low** | Status: **fixed** (#59) | Found: 2026-08-10
File: `pkgs/depiction_io/src/depiction_io/imzml_zip.py:41`

## Symptom

`extract` is documented to "extract the .imzML and .ibd file to the given directory and return
the path to the .imzML file". It returns a path that is either a **directory** or nothing at
all, and the actual extracted file lands somewhere else.

## Why it happens

Two bugs on three lines:

```python
if imzml_filename is None:
    imzml_filename = (
        directory / Path(self.imzml_filename).name
    )  # already includes `directory`
file.extract(self.imzml_filename, directory / imzml_filename)  # joins `directory` again
file.extract(self.ibd_filename, directory / imzml_filename.with_suffix(".ibd"))
return imzml_filename
```

1. `ZipFile.extract(member, path)` treats `path` as the **directory to extract into**, not as
   the destination filename. So a directory named `.../x.imzML` is created and the member is
   written inside it, under its archive-internal path.
2. `imzml_filename` already has `directory` prepended, and then `directory / imzml_filename`
   prepends it a second time. With an absolute `directory` the second join is absorbed by
   `Path` semantics; with a **relative** one it produces `out/out/x.imzML`, so the returned
   `out/x.imzML` does not exist at all.

## Fix

Deletion was considered — the module has zero in-repo callers, is not exported from
`depiction_io/__init__.py`, and the pipeline has a working inline implementation — and
rejected, because `depiction_io` is published for downstream applications that may call it.

`extract` now uses `ZipFile.open` + `shutil.copyfileobj` to write to an explicit destination
and joins `directory` once. `imzml_filename`, if given, is documented as resolved relative to
`directory`. It also raises `ValueError` when the archive holds no single .imzML/.ibd pair,
where it previously died on `Path(None)` with a bare `TypeError` — `_entry_name` has already
logged the real reason by then.

Tests cover both layouts the docstring claims (imzML at the archive root and in a
subdirectory) with a **relative** output directory, which is the case that produced
`out/out/x.imzML` while returning `out/x.imzML`.

## Notes

The class docstring's `TODO when implementing a writer define a standard output format`
suggests this was always a sketch. Nothing in the repo grew to depend on it.
