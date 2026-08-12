Targeted MSI preprocessing pipeline using depiction.

A Snakemake workflow that takes an imzML acquisition and a CSV mass list and produces
calibrated spectra (`calibrated.imzML`), ion images as OME-TIFF and SpatialData
(`images_default.ome.tiff`, `images_default.sd.zarr`) and a set of QC plots. Which of these a
run builds is chosen by `requested_artifacts`; the mapping from artifact to output file is in
`pipeline_config/artifacts_mapping.py`, the parameter model and the shipped presets are in
`pipeline_config/`, and the rules are in `workflow/`.

See the repository README for how to lay out a chunk directory and invoke the pipeline.

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

Three modules that were dead in the same way have been **deleted** rather than documented,
because nothing referenced them and they could not be imported at all: `workflow/experimental.smk`
(never `include:`d by the `Snakefile`, and it redefined `vis_test_mass_shifts`, which
`rules_vis.smk` also defines — so including it would have failed immediately),
`workflow/proc/cluster_stats.py` (imported `__cluster_stats`, a module never committed) and
`depiction/tools/experimental/msi_hdf5.py` (imported `awkward`, declared in no extra).
