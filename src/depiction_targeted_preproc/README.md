Targeted MSI preprocessing pipeline using depiction.

A Snakemake workflow that takes an imzML acquisition and a CSV mass list and produces
calibrated spectra (`calibrated.imzML`), ion images as OME-TIFF and SpatialData
(`images_default.ome.tiff`, `images_default.sd.zarr`) and a set of QC plots. Which of these a
run builds is chosen by `requested_artifacts`; the mapping from artifact to output file is in
`pipeline_config/artifacts_mapping.py`, the parameter model and the shipped presets are in
`pipeline_config/`, and the rules are in `workflow/`.
