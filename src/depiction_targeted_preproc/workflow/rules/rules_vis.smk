rule vis_images:
    input:
        imzml=multiext("{sample}/calibrated",".imzML",".ibd"),
        config="{sample}/pipeline_params.yml",
        mass_list="{sample}/images_{label}_mass_list.csv"
    output:
        hdf5="{sample}/images_{label}.hdf5"
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.images "
        " --imzml-path {input.imzml[0]} --mass-list-path {input.mass_list} "
        " --output-hdf5-path {output.hdf5}"
        " --config-path {input.config}"

rule vis_images_norm:
    input: hdf5="{sample}/images_{label}.hdf5"
    output: hdf5="{sample}/images_{label}_norm.hdf5"
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.images_norm "
        " --input-hdf5-path {input.hdf5} --output-hdf5-path {output.hdf5}"

rule vis_images_ome_tiff:
    input:
        netcdf="{sample}/images_{label}.hdf5",
        raw_metadata="{sample}/raw_metadata.json"
    output:
        ometiff="{sample}/images_{label}.ome.tiff"
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.images_ome_tiff "
        " --input-netcdf-path {input.netcdf} --output-ometiff-path {output.ometiff}"
        " --input-raw-metadata-path {input.raw_metadata}"

rule vis_clustering:
    input:
        netcdf="{sample}/cluster_{label}.hdf5"
    output:
        png="{sample}/cluster_{label}.png"
    shell:
        "python -m depiction_targeted_preproc.workflow.vis.clustering "
        " --input-netcdf-path {input.netcdf} --output-png-path {output.png}"