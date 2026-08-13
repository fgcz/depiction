# The pre-refactor baseline diff

Runs the `depiction_targeted_preproc` pipeline twice on the same acquisition — once in this
tree, once in the tree as it was before the `imzy` migration — and compares the outputs.

**The result, twice: the migration changed nothing in the data.** Run on 2026-08-07 and
re-run in full on 2026-08-12 after the `evaluate_bins` float32 fix, on both fixtures, with
the imzML pairs read under *both* imzy and the pre-refactor parser. Every artifact matched
exactly, with no tolerance — every spectrum's m/z and intensity array, every coordinate,
every image value, every channel name, the calibration coefficients, the OME-TIFF and the
SpatialData zarr. The `.ibd` files are **byte-identical after their 16-byte UUID header**:
`cmp -l` reports exactly 16 differing bytes in a 3.8 GB file.

This directory is how to reproduce it. The differences that do exist are all in the imzML
XML rather than the data, and are catalogued in
[`docs/modules/depiction_io/imzy_backend.md`](../../docs/modules/depiction_io/imzy_backend.md).

Nothing runs it automatically. It needs two environments, a second checkout, and (for the
tonsil) a 1.26 GB fixture, none of which belong in `nox` or in CI.

## What it compares, and why that is one variable

The baseline is **`ed222b3`** ("Spatial dist plot", #31) — the last commit before Phase A,
still on `pyimzml`.

`git diff ed222b3 <this tree>` over the whole `CALIB_IMAGES` chain is imports, type
annotations and comment rewrapping. `rules_proc.smk`, `rules_vis.smk`, `Snakefile` and
`artifacts_mapping.py` are byte-identical; `image/ome_tiff.py` differs by one import line.
So the pipeline either side of the I/O layer is the *same code*, and a difference in the
output can only have come from the reader or the writer.

## Reconstructing the baseline environment

`ed222b3` predates the committed `uv.lock`, so a bare install there resolves ~380 packages at
whatever is current today, and an imzy difference could not be told apart from a scipy bump.
`constraints.txt` is this tree's lock exported to a constraints file, so every shared
dependency installs at the same version in both environments.

```bash
git worktree add ../depiction-baseline ed222b3
uv venv --python 3.13 ../depiction-baseline/.venv
uv pip install --python ../depiction-baseline/.venv/bin/python \
    --constraint system_tests/baseline/constraints.txt -e ../depiction-baseline
```

Two things that file does beyond pinning versions:

- It is exported with `--all-extras`, so it constrains anything the baseline might pull in,
  not only the runtime set.
- It carries `snakemake-invoke @ git+...@e7e3c33` as a URL constraint, which pins the one VCS
  dependency. The baseline's own `pyproject.toml` declares it with **no revision** — the bug
  Phase G found and fixed on `dev` — so installed as written it resolves the git HEAD, where
  `SnakemakeInvoke` is no longer in `__init__.py`, and `process_chunk` fails at import.
  That line is now **hand-maintained**: the current tree vendors `snakemake_invoke` into
  `pkgs/`, so a plain re-export silently drops it. The regeneration comment at the top of
  `constraints.txt` says so; heed it.

Check the result:

```bash
uv pip list --python ../depiction-baseline/.venv/bin/python | grep -iE 'pyimzml|imzy'
```

`pyimzml` present, `imzy` absent. When this was run, the two environments shared 198 packages
at identical versions, and `pyimzml` (plus its `wheezy-template` dependency) was the only
thing unique to the baseline.

## Running it

```bash
uv run python -m system_tests.baseline.run_pipelines \
    --fixture-name mouse_kidney \
    --baseline-python ../depiction-baseline/.venv/bin/python \
    --out-dir /tmp/baseline-diff
```

Both work directories are staged by the same code, from
`system_tests/calibration/configs/`, so the only difference between the runs is the
interpreter. Each gets a `versions.txt` next to it.

Then compare. The imzML pairs are compared **under both readers** — imzy's and the
pre-refactor parser's — because a single reader cannot distinguish "the two files agree" from
"the reader makes the same mistake on both", which is precisely the check Phase E recorded
losing when the second parser was deleted:

```bash
B=/tmp/baseline-diff/mouse_kidney/baseline/work
C=/tmp/baseline-diff/mouse_kidney/current/work

for name in processed calibrated; do
    uv run python -m system_tests.baseline.compare_imzml $B/$name.imzML $C/$name.imzML
    ../depiction-baseline/.venv/bin/python \
        system_tests/baseline/compare_imzml.py $B/$name.imzML $C/$name.imzML
done

uv run python -m system_tests.baseline.compare_outputs $B $C
```

Invoke the baseline one **by path, not with `-m`** — `-m` from the repository root would put
this tree's sources on its `sys.path`, which is the one contamination the comparison cannot
tolerate.

Both scripts exit non-zero on any difference.

## The two fixtures

`mouse_kidney` is public and takes about a minute per tree; `tonsil` is the 1.26 GB FGCZ
acquisition with a real 118-marker panel. Both are described in
[`../fixtures.py`](../fixtures.py); `run_pipelines.py` takes either name and refuses with the
fixture's own "how to obtain" message when the files are absent.

## What is compared

Everything the pipeline writes, by value — never by bytes. Two correct writers disagree on
the imzML UUID, the `IMS:1000091` checksum and the `.ibd` layout by construction, and an
OME-TIFF carries a UUID and a creation date.

| Artifact | Compared as |
|---|---|
| `panels/*.csv`, `raw_metadata.json`, `config/*.yml` | text, exactly — the control, produced by code the migration did not touch |
| `processed.imzML` / `.ibd` | every spectrum's m/z and intensity array, plus dtypes, coordinates and mode. `process_spectra` runs with `steps: []`, so this is a pure read→write round trip |
| `calibrated.imzML` / `.ibd` | the same, after real computation |
| `images_default.hdf5`, `calib_data.hdf5` | values, channel names and foreground mask per `MultiChannelImage` |
| `images_default.ome.tiff` | the same, plus the physical pixel size |
| `images_default.sd.zarr` | the image array and its channel coordinates |

`dtype` is reported next to the values rather than folded into the equality check: an
intensity array that came back `float32` instead of `float64` is a finding even when every
value it can represent is identical.
