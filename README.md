# depiction

This package provides functionality to process and visualize mass-spectrometry imaging data.
Currently, it requires your data to be available in the `imzML` format.

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with three
packages:

- `depiction` (in `src/`): implements the whole functionality to process the data. It also contains
  `depiction_targeted_preproc`, a pipeline that based on some configuration file creates outputs like
  a qc report and .ome.tiff files.
- `depiction_io` (in `pkgs/depiction_io/`): reading and writing MSI data. Everything else talks to
  the protocols in `depiction_io.types` rather than to a file format, so the storage backend can be
  changed without touching the callers. **If reading and writing MSI files is all you need, depend
  on this package rather than on `depiction`** — see
  [`pkgs/depiction_io/README.md`](pkgs/depiction_io/README.md).
- `snakemake_invoke` (in `pkgs/snakemake_invoke/`): a small wrapper for invoking the pipeline's
  Snakemake workflow from Python, vendored from its own repository so that this one has no git
  dependencies. See [`pkgs/snakemake_invoke/README.md`](pkgs/snakemake_invoke/README.md).

## Status

**This repository is dormant as of August 2026.** It is not abandoned mid-thought — it was
deliberately brought to a coherent, reproducible state and then left there. Nobody is actively
developing it, and there is no maintainer to reach out to. Treat it as an archive you can run,
not as a project you can file a bug against.

What is supported, in the sense that it is tested, exercised end to end and known to work:

- **`depiction_io`** — reading and writing MSI data. It has its own test suite, a protocol
  conformance test per backend, and a differential corpus. If reading and writing imzML is all
  you need, depend on this and ignore the rest.
- **The targeted preprocessing pipeline** (`depiction_targeted_preproc`) — calibration, ion
  images and QC plots, described under [Running the pipeline](#running-the-pipeline). It is
  covered by [system tests](system_tests/README.md) against two real acquisitions, and its
  output was pinned against the pre-migration implementation in
  [`docs/refactoring/baseline-diff.md`](docs/refactoring/baseline-diff.md).

What is **not** supported, and will fail if you call it:

- **The clustering *tools*** — `depiction.tools.clustering` and `depiction.clustering`. They
  raise immediately, and the sampling geometry underneath them is a research question rather
  than a bug with a known fix. Note that this does *not* include the pipeline's own
  `workflow/proc/cluster_kmeans.py` and `workflow/proc/cluster_hdbscan.py`, which are separate
  code, work, and are covered by tests.
- **`depiction_cluster_sandbox`** — needs a data directory that no longer exists and a
  dependency declared in no extra.
- **`src/depiction_targeted_preproc/workflow/exp/`** — one-off experiment scripts, kept for
  reference, reachable from no artifact.

The reasoning behind the dormancy decision, what was migrated and what was deliberately left
undone, is in [`docs/refactoring/ROADMAP.md`](docs/refactoring/ROADMAP.md).

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

Add `--extra findmf` if you need the FindMF peak picker (the `dev_presets` and
`dev_no_calibration` pipeline presets use it). It is optional because `findmfpy` is a C++
extension that ships wheels only for CPython 3.13 on macOS arm64 and x86-64 Linux; anywhere
else, installing it means compiling it.
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

`nox -s docs` builds the Sphinx documentation with warnings treated as errors, so a broken autodoc
reference fails the build rather than silently dropping a page. It is not part of the default run:
intersphinx fetches remote inventories, so an upstream site moving one would fail a bare `nox` on a
commit that changed nothing. `nox -s licensecheck` is non-default for the same reason. CI runs both
in their own job. The slow end-to-end `system_tests` session is not a default either; see
[system_tests/README.md](system_tests/README.md).

However, you can also run the tests with `pytest` or from your IDE if you are in the virtual environment.

## Running the pipeline

The targeted preprocessing pipeline is a Snakemake workflow, but you do not invoke `snakemake`
yourself. The entry point is `process_chunk`, which takes one **chunk directory** — a directory
holding one acquisition and the configuration for it — and builds whatever that configuration
asks for:

```bash
python -m depiction_targeted_preproc.app_interface.process_chunk /path/to/work/my_sample
```

It is deliberately not a console script: it is called this way by the application layer that
used to wrap it, and by [`system_tests/`](system_tests/README.md).

### The chunk directory

The directory's **name is the sample name**, and its parent is what Snakemake is pointed at as
the working directory — so the chunk directory is always one level down, never the top of the
tree. Before the run it holds only inputs:

```
work/
└── my_sample/                        <- pass this path to process_chunk
    ├── params.yml                    <- what to build, and with which preset
    ├── raw.imzML                     <- the acquisition
    ├── raw.ibd
    └── panels/
        └── unstandardized_full.csv   <- the mass list
```

A `raw.imzML.zip` containing both members is accepted in place of the `raw.imzML`/`raw.ibd`
pair; a rule unpacks it first.

The mass list needs a label column and a mass column. Their headers are normalised, so
`marker`/`label`, and `pc-mt (m+h)+`/`mass`/`m/z`, are all recognised; an optional `type` column
defaults to `target`, and only `target` rows are used for calibration.
[`system_tests/panels/mouse_kidney.csv`](system_tests/panels/mouse_kidney.csv) is a two-column
example.

### `params.yml`

```yaml
config_preset: dev_presets   # a file in pipeline_config/config_presets/, without the .yml
requested_artifacts:         # what to build; see the table below
  - CALIB_IMAGES
  - CALIB_QC
n_jobs: 10                   # optional, defaults to 10
```

The model is `Params` in
[`pipeline/prepare_params.py`](src/depiction_targeted_preproc/pipeline/prepare_params.py). It
also accepts a `mass_list_id`, which nothing reads — it is a vestige of the application layer
that used to fetch panels by id, and can be omitted.

The preset supplies the actual processing and calibration configuration. Four ship with the
package, in
[`pipeline_config/config_presets/`](src/depiction_targeted_preproc/pipeline_config/config_presets):
`dev_presets`, `dev_presets_timstof`, `dev_no_calibration` and `dev_no_proc`. A rule at the head
of the workflow expands `params.yml` plus the named preset into a `pipeline_params.yml` next to
it; writing that file yourself instead skips the preset lookup entirely, which is how the system
tests pin an exact configuration without adding a preset for it.

Note that `dev_presets` and `dev_no_calibration` select the FindMF peak picker, so they need the
optional `findmf` extra described above.

### Requested artifacts

`requested_artifacts` is a list of `PipelineArtifact` values, and it is the only thing that
decides how much of the rule graph runs. The mapping from artifact to output file is
[`pipeline_config/artifacts_mapping.py`](src/depiction_targeted_preproc/pipeline_config/artifacts_mapping.py):

| Artifact | Produces |
|---|---|
| `CALIB_IMZML` | `calibrated.imzML` / `.ibd` |
| `CALIB_IMAGES` | `images_default.ome.tiff`, `images_default.sd.zarr` |
| `CALIB_QC` | seven QC plots under `qc/`, plus `qc/calibration_model_coefficients.hdf5` |
| `RAW_TIC` | `tic_image.ome.tiff`, `tic_image.sd.zarr` |
| `DEBUG` | additional QC plots, a k-means clustering and a normalised image stack |

Outputs are written into the chunk directory alongside the inputs, and every requested file is
also collected into `outputs/<sample_name>.zip`.

### Editing the workflow

The rules live in
[`workflow/`](src/depiction_targeted_preproc/workflow), split by stage —
`rules/rules_proc.smk` for processing and calibration, `rules_vis.smk` for images,
`rules_qc.smk` for the QC plots — and `workflow/Snakefile` is the list of what is included.
Every rule is a thin wrapper: it declares inputs and outputs and shells out to
`python -m depiction_targeted_preproc.workflow.<stage>.<script>`, so the code you want to change
is almost always the script, not the rule.

Adding an output means three edits, in this order: the script, a rule that produces its file,
and an entry in `ARTIFACT_FILES_MAPPING` so that some `PipelineArtifact` asks for it. Nothing
runs that no artifact requests — a rule with no path to a requested file is simply never
scheduled, which is worth knowing before debugging why a new rule "does nothing".

`src/depiction_targeted_preproc/README.md` lists the rules that are known to be dead, so you
do not copy one as a template.

### A worked example

[`system_tests/`](system_tests/README.md) is the end-to-end recipe, and it runs on a public
59 MB acquisition that a fresh clone can fetch:

```bash
uv run python -m tests.real_data.fetch mouse_kidney
nox -s system_tests
```

[`system_tests/baseline/run_pipelines.py`](system_tests/baseline/run_pipelines.py) is the
shortest readable example of staging a chunk directory from scratch — its `stage()` function is
the layout above, in twelve lines of code.

## Geometry Conventions

TODO these are not used consistently everywhere yet

### Dimension names

- (2D) Points: (x, y)
- (2D) Images: (y, x, c)
- Sparse images: (i, c)
- Coordinates: (i, d) and each row corresponds to point ordering (i.e. (x, y))

TODO: y-axis direction, xarray conventions (dims, coords, etc.)
