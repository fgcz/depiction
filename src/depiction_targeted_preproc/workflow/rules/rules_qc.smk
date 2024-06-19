rule qc_table_marker_distances_baseline:
    input:
        imzml=multiext("{sample}/corrected.peaks",".imzML",".ibd"),
        mass_list="{sample}/mass_list.visualization.csv",
    output:
        table="{sample}/qc/table_marker_distances_baseline.parquet"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.table_marker_distances"
        " --imzml-peaks {input.imzml[0]} --mass-list {input.mass_list}"
        " --output-table {output.table}"

rule qc_table_marker_distances_calib:
    input:
        imzml_peaks="{sample}/calibrated.imzML",
        mass_list="{sample}/mass_list.visualization.csv",
    output:
        table="{sample}/qc/table_marker_distances_calib.parquet"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.table_marker_distances"
        " --imzml-peaks {input.imzml_peaks} --mass-list {input.mass_list}"
        " --output-table {output.table}"


rule qc_plot_marker_presence:
    input:
        table_marker_distances_baseline="{sample}/qc/table_marker_distances_baseline.parquet",
        table_marker_distances_calib="{sample}/qc/table_marker_distances_calib.parquet"
    output:
        pdf="{sample}/qc/plot_marker_presence.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_marker_presence"
        " --table-marker-distances-baseline {input.table_marker_distances_baseline}"
        " --table-marker-distances-calib {input.table_marker_distances_calib}"
        " --output-pdf {output.pdf}"


rule qc_plot_marker_presence_cv:
    input:
        image_hdf5="{sample}/images_default.hdf5"
    output:
        pdf="{sample}/qc/plot_marker_presence_cv.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_marker_presence_cv"
        " --image-hdf5 {input.image_hdf5}"
        " --output-pdf {output.pdf}"


rule qc_plot_peak_density_combined:
    input:
        table_marker_distances_baseline="{sample}/qc/table_marker_distances_baseline.parquet",
        table_marker_distances_calib="{sample}/qc/table_marker_distances_calib.parquet"
    output:
        pdf="{sample}/qc/plot_peak_density_combined.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_peak_density"
        " --table-marker-distances-baseline {input.table_marker_distances_baseline}"
        " --table-marker-distances-calib {input.table_marker_distances_calib}"
        " --no-grouped"
        " --output-pdf {output.pdf}"


rule qc_plot_peak_density_grouped:
    input:
        table_marker_distances_baseline="{sample}/qc/table_marker_distances_baseline.parquet",
        table_marker_distances_calib="{sample}/qc/table_marker_distances_calib.parquet"
    output:
        pdf="{sample}/qc/plot_peak_density_grouped.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_peak_density"
        " --table-marker-distances-baseline {input.table_marker_distances_baseline}"
        " --table-marker-distances-calib {input.table_marker_distances_calib}"
        " --grouped"
        " --output-pdf {output.pdf}"


# TODO this plot has the advantage of showing 3 mass ranges, which could indicate some problems which are missed
#      when rendering just one plot
rule qc_plot_calibration_map:
    input:
        calib_data="{sample}/calib_data.hdf5",
        mass_list="{sample}/mass_list.visualization.csv",
    output:
        pdf="{sample}/qc/plot_calibration_map.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_calibration_map"
        " --calib-data {input.calib_data} --mass-list {input.mass_list}"
        " --output-pdf {output.pdf}"


rule qc_plot_calibration_map_v2:
    input:
        mass_shifts="{sample}/test_mass_shifts.hdf5",
    output:
        pdf="{sample}/qc/plot_calibration_map_v2.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_calibration_map_v2"
        " --input-mass-shifts {input.mass_shifts}"
        " --output-pdf {output.pdf}"

rule qc_plot_sample_spectra_before_after:
    input:
        imzml_baseline="{sample}/peaks.imzML",
        imzml_calib="{sample}/calibrated.imzML",
        mass_list="{sample}/mass_list.visualization.csv",
    output:
        pdf="{sample}/qc/plot_sample_spectra_before_after.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_sample_spectra_before_after"
        " --imzml-baseline {input.imzml_baseline} --imzml-calib {input.imzml_calib}"
        " --mass-list {input.mass_list}"
        " --output-pdf {output.pdf}"

#TODO
rule qc_plot_spectra_for_marker:
    input:
        marker_surrounding_baseline="{sample}/qc/table_marker_distances_baseline.parquet",
        marker_surrounding_calib="{sample}/qc/table_marker_distances_calib.parquet",
    output:
        pdf="{sample}/qc/plot_spectra_for_marker.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_spectra_for_marker"
        " --marker-surrounding-baseline {input.marker_surrounding_baseline}"
        " --marker-surrounding-calib {input.marker_surrounding_calib}"
        " --output-pdf {output.pdf}"

rule qc_plot_peak_counts:
    input:
        imzml=multiext("{sample}/corrected.peaks",".imzML",".ibd"),
        config="{sample}/pipeline_params.yml",
    output:
        pdf="{sample}/qc/plot_peak_counts.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_peak_counts"
        " --config-path {input.config} --imzml-peaks {input.imzml[0]}"
        " --output-pdf {output.pdf}"

rule qc_plot_scan_direction:
    input:
        imzml=multiext("{sample}/corrected.peaks",".imzML",".ibd"),
    output:
        pdf="{sample}/qc/plot_scan_direction.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_scan_direction"
        " --input-imzml-path {input.imzml[0]}"
        " --output-pdf {output.pdf}"
