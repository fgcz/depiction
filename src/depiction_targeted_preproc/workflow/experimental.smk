version: "3"

# TODO it should be more clear which is baseline+profile and which is baseline+picked because currently this is a bit
#      ambiguous. in principle we should structure the pipeline in a way, that in subsequent steps it is transparent
#      which of these two it is.

include: "rules/rules_proc.smk"

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

rule qc_peak_counts:
    input:
        imzml_peaks="{sample}/peaks.imzML",
    output:
        pdf="{sample}/qc/peak_counts.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.peak_counts"
        " --imzml-peaks {input.imzml_peaks} --output-table {output.table}"


rule qc_table_marker_distances_baseline:
    input:
        imzml_peaks="{sample}/peaks.imzML",
        mass_list="{sample}/images_default_mass_list.csv",
    output:
        table="{sample}/qc/table_marker_distances_baseline.parquet"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.table_marker_distances"
        " --imzml-peaks {input.imzml_peaks} --mass-list {input.mass_list}"
        " --output-table {output.table}"

rule qc_table_marker_distances_calib:
    input:
        imzml_peaks="{sample}/calibrated.imzML",
        mass_list="{sample}/images_default_mass_list.csv",
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


rule qc_plot_calibration_map:
    input:
        calib_data="{sample}/calib_data.hdf5",
        mass_list="{sample}/images_default_mass_list.csv",
    output:
        pdf="{sample}/qc/plot_calibration_map.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_calibration_map"
        " --calib-data {input.calib_data} --mass-list {input.mass_list}"
        " --output-pdf {output.pdf}"


rule qc_plot_sample_spectra_before_after:
    input:
        imzml_baseline="{sample}/peaks.imzML",
        imzml_calib="{sample}/calibrated.imzML",
        mass_list="{sample}/images_default_mass_list.csv",
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
        imzml_peaks="{sample}/peaks.imzML",
        config="{sample}/pipeline_params.yml",
    output:
        pdf="{sample}/qc/plot_peak_counts.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_peak_counts"
        " --config-path {input.config} --imzml-peaks {input.imzml_peaks}"
        " --output-pdf {output.pdf}"



#rule vis_calib_heatmap:
#    input:
#        calib_data="{sample}/calib_data.hdf5",
#    output:
#        hdf5="{sample}/images_calib_heatmap.hdf5"
#    shell:
#        "python -m depiction_targeted_preproc.workflow.vis.calib_heatmap "
#        " --input-calib-data-path {input.calib_data}"
#        " --output-hdf5-path {output.hdf5}"
