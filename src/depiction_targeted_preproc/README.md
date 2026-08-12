Targeted MSI preprocessing pipeline using depiction.

A Snakemake workflow that takes an imzML acquisition and a CSV mass list and produces
calibrated spectra (`calibrated.imzML`), ion images as OME-TIFF and SpatialData
(`images_default.ome.tiff`, `images_default.sd.zarr`) and a set of QC plots. Which of these a
run builds is chosen by `requested_artifacts`; the mapping from artifact to output file is in
`pipeline_config/artifacts_mapping.py`, the parameter model and the shipped presets are in
`pipeline_config/`, and the rules are in `workflow/`.

## Running it

You do not invoke `snakemake` yourself. The entry point is `process_chunk`, which takes one
**chunk directory** — a directory holding one acquisition and the configuration for it — and
builds whatever that configuration asks for:

```bash
python -m depiction_targeted_preproc.app_interface.process_chunk /path/to/work/my_sample
```

It is deliberately not a console script: it is called this way by the application layer that used
to wrap it, and by [`system_tests/`](../../system_tests/README.md).

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
[`system_tests/panels/mouse_kidney.csv`](../../system_tests/panels/mouse_kidney.csv) is a
two-column example.

### `params.yml`

```yaml
config_preset: dev_presets   # a file in pipeline_config/config_presets/, without the .yml
requested_artifacts:         # what to build; see the table below
  - CALIB_IMAGES
  - CALIB_QC
n_jobs: 10                   # optional, defaults to 10
```

The model is `Params` in [`pipeline/prepare_params.py`](pipeline/prepare_params.py). It also
accepts a `mass_list_id`, which nothing reads — it is a vestige of the application layer that
used to fetch panels by id, and can be omitted.

The preset supplies the actual processing and calibration configuration. Four ship with the
package, in [`pipeline_config/config_presets/`](pipeline_config/config_presets): `dev_presets`,
`dev_presets_timstof`, `dev_no_calibration` and `dev_no_proc`. A rule at the head of the workflow
expands `params.yml` plus the named preset into a `pipeline_params.yml` next to it; writing that
file yourself instead skips the preset lookup entirely, which is how the system tests pin an
exact configuration without adding a preset for it.

`dev_presets` and `dev_no_calibration` select the FindMF peak picker, so they need the optional
`findmf` extra — see the repository README for why it is optional.

### Requested artifacts

`requested_artifacts` is a list of `PipelineArtifact` values, and it is the only thing that
decides how much of the rule graph runs. The mapping from artifact to output file is
[`pipeline_config/artifacts_mapping.py`](pipeline_config/artifacts_mapping.py):

| Artifact | Produces |
|---|---|
| `CALIB_IMZML` | `calibrated.imzML` / `.ibd` |
| `CALIB_IMAGES` | `images_default.ome.tiff`, `images_default.sd.zarr` |
| `CALIB_QC` | seven QC plots under `qc/`, plus `qc/calibration_model_coefficients.hdf5` |
| `RAW_TIC` | `tic_image.ome.tiff`, `tic_image.sd.zarr` |
| `DEBUG` | additional QC plots, a k-means clustering and a normalised image stack |

`DEBUG` deliberately does not include the hdbscan clustering, even though
`proc_cluster_hdbscan` works: `hdbscan` is declared only in the `dev` extra and publishes no
manylinux wheel, so requesting it from an artifact would make every clean Linux install compile
it.

Outputs are written into the chunk directory alongside the inputs, and every requested file is
also collected into `outputs/<sample_name>.zip`.

### A worked example

[`system_tests/`](../../system_tests/README.md) is the end-to-end recipe, and it runs on a public
59 MB acquisition that a fresh clone can fetch:

```bash
uv run python -m tests.real_data.fetch mouse_kidney
nox -s system_tests
```

[`system_tests/baseline/run_pipelines.py`](../../system_tests/baseline/run_pipelines.py) is the
shortest readable example of staging a chunk directory from scratch — its `stage()` function is
the layout above, in a dozen lines of code.

## Editing the workflow

The rules live in [`workflow/`](workflow), split by stage — `rules/rules_proc.smk` for processing
and calibration, `rules_vis.smk` for images, `rules_qc.smk` for the QC plots — and
`workflow/Snakefile` is the list of what is included. Every rule is a thin wrapper: it declares
inputs and outputs and shells out to `python -m depiction_targeted_preproc.workflow.<stage>.<script>`,
so the code you want to change is almost always the script, not the rule.

Adding an output means three edits, in this order: the script, a rule that produces its file,
and an entry in `ARTIFACT_FILES_MAPPING` so that some `PipelineArtifact` asks for it. Nothing
runs that no artifact requests — a rule with no path to a requested file is simply never
scheduled, which is worth knowing before debugging why a new rule "does nothing".

Before copying a rule as a template, check it against the list below.

## Known-dead rules — do not use

These are in the tree, are loaded by the `Snakefile`, and **cannot run**. None is reachable
from any `PipelineArtifact`, so a normal run never touches them; they only bite someone who
asks for their output by filename, or who copies one as a template. Each fails at DAG
construction with a missing input rather than producing anything wrong.

| Where | Why it cannot run |
|---|---|
| rule `qc_plot_sample_spectra_before_after`, `workflow/rules/rules_qc.smk` | Wants `{sample}/peaks.imzML`. No rule anywhere produces that file — the only mention of it in the workflow is this rule's own `input`. |
| `workflow/rules/rules_simulate.smk`, rule `simulate_generate_imzml` onwards | Wants `{sample}_sim/full.csv`. The rule meant to produce it, `simulate_create_mass_list`, writes `{sample}_sim/unstandardize_full.csv` instead, and no standardization rule bridges the two: `panel_standardized_full` reads `{sample}/panels/unstandardized_full.csv`, a different path under a different directory. The whole simulate chain is broken at that seam. |

`workflow/exp/` is a different case: those are one-off experiment scripts that are not wired
to any rule at all. They are kept as reference for what was tried, not as something to run.
One of them, `exp/compare_cluster_stats.py`, reads `cluster_default_stats_*.csv`, which nothing
in the tree produces any more — the rule that wrote it is among the deletions below.

Three modules that were dead in the same way have been **deleted** rather than documented,
because nothing referenced them and they could not be imported at all: `workflow/experimental.smk`
(never `include:`d by the `Snakefile`, and it redefined `vis_test_mass_shifts`, which
`rules_vis.smk` also defines — so including it would have failed immediately),
`workflow/proc/cluster_stats.py` (imported `__cluster_stats`, a module never committed) and
`depiction/tools/experimental/msi_hdf5.py` (imported `awkward`, declared in no extra).
