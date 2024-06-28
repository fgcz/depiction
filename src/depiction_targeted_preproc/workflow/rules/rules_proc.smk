rule proc_correct_baseline:
    input:
        imzml=multiext("{sample}/raw", ".imzML", ".ibd"),
        config="{sample}/pipeline_params.yml",
    output:
        imzml=temp(multiext("{sample}/corrected.original", ".imzML", ".ibd")),
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.correct_baseline "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} "
        " --output-imzml-path {output.imzml[0]}"


rule proc_pick_peaks:
    input:
        imzml=multiext("{sample}/corrected.original", ".imzML", ".ibd"),
        config="{sample}/pipeline_params.yml",
    output:
        imzml=multiext("{sample}/corrected.peaks", ".imzML", ".ibd"),
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.pick_peaks "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} "
        " --output-imzml-path {output.imzml[0]}"


# TODO currently needed for peak picked data, e.g. from timsTOF
# rule proc_pick_peaks:
#    input:
#        imzml=multiext("{sample}/corrected.original",".imzML",".ibd"),
#        config="{sample}/pipeline_params.yml",
#    output:
#        imzml=multiext("{sample}/corrected.peaks_all",".imzML",".ibd"),
#    shell:
#        "python -m depiction_targeted_preproc.workflow.proc.pick_peaks "
#        " --input-imzml-path {input.imzml[0]} --config-path {input.config} "
#        " --output-imzml-path {output.imzml[0]}"
#
#
## TODO very experimental to be removed later again
# rule proc_filter_peaks:
#    input:
#        imzml=multiext("{sample}/corrected.peaks_all",".imzML",".ibd"),
#    output:
#        imzml=multiext("{sample}/corrected.peaks",".imzML",".ibd"),
#    shell:
#        "python -m depiction_targeted_preproc.workflow.proc.filter_peaks "
#        " --input-imzml-path {input.imzml[0]} "
#        " --output-imzml-path {output.imzml[0]}"


# TODO this should be solved more efficiently in the future, but for now it is solved by calling the script twice
rule proc_calibrate_remove_global_shift:
    input:
        imzml=multiext("{sample}/corrected.peaks", ".imzML", ".ibd"),
        config="{sample}/pipeline_params.yml",
        #mass_list="{sample}/mass_list.standards.csv",
        mass_list="{sample}/mass_list.calibration.csv",
    output:
        imzml=temp(multiext("{sample}/calibrated.tmp", ".imzML", ".ibd")),
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.calibrate "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} --mass-list-path {input.mass_list} "
        " --use-global-constant-shift"
        " --output-imzml-path {output.imzml[0]}"


rule proc_calibrate_actual:
    input:
        imzml=multiext("{sample}/calibrated.tmp", ".imzML", ".ibd"),
        config="{sample}/pipeline_params.yml",
        mass_list="{sample}/mass_list.calibration.csv",
    output:
        imzml=multiext("{sample}/calibrated", ".imzML", ".ibd"),
        calib_data="{sample}/calib_data.hdf5",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.calibrate "
        " --input-imzml-path {input.imzml[0]} --config-path {input.config} --mass-list-path {input.mass_list} "
        " --output-imzml-path {output.imzml[0]} --output-calib-data-path {output.calib_data}"


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


rule proc_cluster_stats:
    input:
        netcdf="{sample}/cluster_default_{variant}.hdf5",
    output:
        csv="{sample}/cluster_default_stats_{variant}.csv",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.cluster_stats"
        " --input-netcdf-path {input.netcdf} --output-csv-path {output.csv}"


rule proc_mass_list_preparation:
    input:
        csv="{sample}/mass_list.raw.csv",
    output:
        calibration_csv="{sample}/mass_list.calibration.csv",
        standards_csv="{sample}/mass_list.standards.csv",
        visualization_csv="{sample}/mass_list.visualization.csv",
        visualization_mini_csv="{sample}/mass_list.visualization_mini.csv",
    shell:
        "python -m depiction_targeted_preproc.workflow.proc.mass_list_preparation"
        " --input-csv-path {input.csv}"
        " --out-calibration-csv-path {output.calibration_csv}"
        " --out-standards-csv-path {output.standards_csv}"
        " --out-visualization-csv-path {output.visualization_csv}"
        " --out-visualization-mini-csv-path {output.visualization_mini_csv}"
