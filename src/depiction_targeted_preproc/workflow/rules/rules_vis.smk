rule vis_images:
    input:
        imzml=multiext("{sample}/calibrated", ".imzML", ".ibd"),
        config="{sample}/pipeline_params.yml",
        mass_list="{sample}/mass_list.visualization.csv",
    output:
        hdf5="{sample}/images_default.hdf5",
    # TODO how can i pass n-jobs nicely here
    shell:
        "python -m depiction.tools.cli.cli_generate_ion_images"
        " --imzml-path {input.imzml[0]} --mass-list-path {input.mass_list}"
        " --output-hdf5-path {output.hdf5} --n-jobs 10"


rule vis_images_norm:
    input:
        hdf5="{sample}/images_{label}.hdf5",
    output:
        hdf5="{sample}/images_{label}_norm.hdf5",
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.images_norm "
        " --input-hdf5-path {input.hdf5} --output-hdf5-path {output.hdf5}"


rule vis_images_ome_tiff:
    input:
        netcdf="{sample}/images_{label}.hdf5",
        raw_metadata="{sample}/raw_metadata.json",
    output:
        ometiff="{sample}/images_{label}.ome.tiff",
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.images_ome_tiff "
        " --input-netcdf-path {input.netcdf} --output-ometiff-path {output.ometiff}"
        " --input-raw-metadata-path {input.raw_metadata}"


rule vis_clustering:
    input:
        netcdf="{sample}/cluster_{label}.hdf5",
    output:
        png="{sample}/cluster_{label}.png",
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.clustering "
        " --input-netcdf-path {input.netcdf} --output-png-path {output.png}"


rule vis_test_mass_shifts:
    input:
        calib_hdf5="{sample}/calib_data.hdf5",
        config="{sample}/config/proc_calibrate.yml",
        mass_list="{sample}/mass_list.calibration.csv",
    output:
        hdf5="{sample}/test_mass_shifts.hdf5",
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.test_mass_shifts "
        " --calib-hdf5-path {input.calib_hdf5} --mass-list-path {input.mass_list} "
        " --config-path {input.config}"
        " --output-hdf5-path {output.hdf5}"
