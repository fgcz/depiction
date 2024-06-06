rule proc_correct_baseline:
    input:
        imzml=multiext("{sample}/raw",".imzML",".ibd"),
        config="{sample}/pipeline_params.yml",
    output:
        imzml=temp(multiext("{sample}/corrected.original",".imzML",".ibd"))
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.correct_baseline "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} "
        " --output-imzml-path {output.imzml[0]}"


rule proc_pick_peaks:
    input:
        imzml=multiext("{sample}/corrected.original",".imzML",".ibd"),
        config="{sample}/pipeline_params.yml",
    output:
        imzml=multiext("{sample}/corrected.peaks",".imzML",".ibd"),
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.pick_peaks "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} "
        " --output-imzml-path {output.imzml[0]}"


# TODO this should be solved more efficiently in the future, but for now it is solved by calling the script twice
rule proc_calibrate_remove_global_shift:
    input:
        imzml=multiext("{sample}/corrected.peaks",".imzML",".ibd"),
        config="{sample}/pipeline_params.yml",
        mass_list="{sample}/images_default_mass_list.csv",
    output:
        imzml=temp(multiext("{sample}/calibrated.tmp",".imzML",".ibd")),
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.calibrate "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} --mass-list-path {input.mass_list} "
        " --use-global-constant-shift"
        " --output-imzml-path {output.imzml[0]}"

rule proc_calibrate_actual:
    input:
        imzml=multiext("{sample}/calibrated.tmp",".imzML",".ibd"),
        config="{sample}/pipeline_params.yml",
        mass_list="{sample}/images_default_mass_list.csv",
    output:
        imzml=multiext("{sample}/calibrated",".imzML",".ibd"),
        calib_data="{sample}/calib_data.hdf5",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.calibrate "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} --mass-list-path {input.mass_list} "
        " --output-imzml-path {output.imzml[0]} --output-calib-data-path {output.calib_data}"

rule proc_export_raw_metadata:
    input:
        imzml="{sample}/raw.imzML",
    output:
        json="{sample}/raw_metadata.json"
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.export_raw_metadata "
        " --input-imzml-path {input.imzml} --output-json-path {output.json}"

rule proc_cluster_kmeans:
    input:
        netcdf="{sample}/images_default.hdf5"
    output:
        netcdf="{sample}/cluster_default_kmeans.hdf5"
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.cluster_kmeans "
        " --input-netcdf-path {input.netcdf} --output-netcdf-path {output.netcdf}"

rule proc_cluster_stats_kmeans:
    input:
        netcdf="{sample}/cluster_default_kmeans.hdf5"
    output:
        csv="{sample}/cluster_default_stats_kmeans.csv"
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.cluster_stats"
        " --input-netcdf-path {input.netcdf} --output-csv-path {output.csv}"
