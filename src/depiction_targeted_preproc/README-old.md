## Parameters

| parameter    | description                                   | always needed                                                                |
| ------------ | --------------------------------------------- | ---------------------------------------------------------------------------- |
| panel_csv    | the mass list                                 | yes, all targeted pipelines will require the mass list in one way or another |
| baseline_adj | which baseline adjustment should be performed | yes, with default                                                            |
| vis_mass_tol | the mass tolerance for visualization          | no, only when generating images                                              |

## Possible outputs

Output groups

| name         | files                                   | description                                                                                          | depends on |
| ------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------- |
| calib-imzml  | calibrated.imzML, calibrated.ibd        | the calibrated imzML file                                                                            | calib-qc   |
| calib-images | images.hdf5                             | the calibrated images as spatialdata object (TODO check if the mass list would need a separate file) | calib-qc   |
| calib-qc     | calibration_qc.pdf, calibration_q5.hdf5 | the calibration QC                                                                                   |

TODO lets say i want to create multiple different types of images from the same
calib-imzml, then this should somehow be possible to configure

- but to do so without a total mess would be potentially difficult

For calib-images there are some questions that need to be addressed:

- baseline adjusted vs non-adjusted?
- normalized vs non-normalized (I would vote to not normalize)

TODO the problem is that it would be more useful to define output groups, since
usually more than one file will be involved

| filename               | description                                               |
| ---------------------- | --------------------------------------------------------- |
| calibrated.imzML       | The calibrated .imzML + .ibd files for further processing |
| calibrated_images.hdf5 | Spatial data object containing the calibrated images.     |

## Unsorted notes

I think this pipeline should be somewhat generic in the outputs that it will
generate. These can then be selected by specifying a parameter, that will
trigger only the requested targets. In particular imzML data can be fairly large
very quick, so we need to have a good idea how to deal with that in practice.

Possible artifiacts

Steps performed:

- Load imzml
- Compute calibration
  - Output: new imzML file
- TODO but how to provide the various intermediaries (configurable target list?)

e.g.

request artifacts:

- `calibrated.imzML`
- `calibrated.imzML channels.hdf5`
- `channels.hdf5`

and under the hood it calls a snakemake workflow that then creates the requested
data, this will allow us to have a pipeline with high flexibility without having
to create way too many applications

in particular, it would even be compatible with a dataset based workflow, since
there could simply be a column that would specify the desired artfacts
