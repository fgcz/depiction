rule process_spectra_config:
    input:
        config="{sample}/pipeline_params.yml",
    output:
        config="{sample}/config/process_spectra.yml",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.process_spectra_config"
        " --input-config {input.config} --output-config {output.config}"


rule process_spectra_run:
    input:
        imzml=multiext("{sample}/raw", ".imzML", ".ibd"),
        config="{sample}/config/process_spectra.yml",
    output:
        imzml=multiext("{sample}/processed", ".imzML", ".ibd"),
    # The tool parallelises internally over `n_jobs`, which is the same number this workflow
    # is given as `--cores`. Claiming all of them stops snakemake from starting another heavy
    # rule alongside it and oversubscribing the machine.
    threads: workflow.cores
    shell:
        "python -m depiction.tools.process_spectra"
        " --config-file {input.config} --input-imzml-file {input.imzml[0]} --output-imzml-file {output.imzml[0]}"


rule proc_calibrate_config:
    input:
        config="{sample}/pipeline_params.yml",
    output:
        config="{sample}/config/proc_calibrate.yml",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.calibrate_config"
        " --input-config {input.config} --output-config {output.config}"


rule proc_calibrate:
    input:
        imzml=multiext("{sample}/processed", ".imzML", ".ibd"),
        config="{sample}/config/proc_calibrate.yml",
        mass_list="{sample}/panels/calibration.csv",
    output:
        imzml=multiext("{sample}/calibrated", ".imzML", ".ibd"),
        calib_data="{sample}/calib_data.hdf5",
    # As above: parallelises internally over `n_jobs`.
    threads: workflow.cores
    shell:
        "python -m depiction.tools.calibrate "
        " run-config --config {input.config} "
        " --input-imzml {input.imzml[0]} --input-mass-list {input.mass_list}"
        " --output-imzml {output.imzml[0]} --output-calib-data {output.calib_data}"


rule proc_export_raw_metadata:
    input:
        imzml="{sample}/raw.imzML",
    output:
        json="{sample}/raw_metadata.json",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.export_raw_metadata "
        " --input-imzml-path {input.imzml} --output-json-path {output.json}"


rule proc_cluster_kmeans:
    input:
        netcdf="{sample}/images_default.hdf5",
    output:
        netcdf="{sample}/cluster_default_kmeans.hdf5",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.cluster_kmeans "
        " --input-netcdf-path {input.netcdf} --output-netcdf-path {output.netcdf}"


rule proc_cluster_hdbscan:
    input:
        netcdf="{sample}/images_default.hdf5",
    output:
        netcdf="{sample}/cluster_default_hdbscan.hdf5",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.cluster_hdbscan "
        " --input-netcdf-path {input.netcdf} --output-netcdf-path {output.netcdf}"
