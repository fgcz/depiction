version: "3"


include: "rules/rules_proc.smk"
include: "rules/rules_vis.smk"
include: "rules/rules_qc.smk"
include: "rules/rules_simulate.smk"


exp_variants = ["chem_noise", "mass_cluster", "reg_shift"]


rule exp_compare_cluster_stats:
    input:
        csv=expand("{{sample}}/{exp_variant}/cluster_default_stats_hdbscan.csv", exp_variant=exp_variants),
    output:
        pdf="{sample}/exp_compare_cluster_stats.pdf",
    shell:
        "python -m depiction_targeted_preproc.workflow.exp.compare_cluster_stats"
        " {input.csv}"
        " --output-pdf {output}"


rule exp_plot_compare_peak_density:
    input:
        tables_marker_distance=expand(
            "{{sample}}/{exp_variant}/qc/table_marker_distances_calib.parquet", exp_variant=exp_variants
        ),
        table_marker_distance_uncalib="{sample}/reg_shift/qc/table_marker_distances_baseline.parquet",
    output:
        pdf="{sample}/exp_plot_compare_peak_density.pdf",
    shell:
        "python -m depiction_targeted_preproc.workflow.exp.plot_compare_peak_density"
        " {input.tables_marker_distance}"
        " --table-marker-distance-uncalib {input.table_marker_distance_uncalib}"
        " --output-pdf {output.pdf}"


# for the poster:
rule qc_table_marker_distances_baseline_mini:
    input:
        imzml=multiext("{sample}/corrected.peaks", ".imzML", ".ibd"),
        mass_list="{sample}/mass_list.visualization_mini.csv",
    output:
        table="{sample}/qc/table_marker_distances_baseline_mini.parquet",
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.table_marker_distances"
        " --imzml-peaks {input.imzml[0]} --mass-list {input.mass_list}"
        " --output-table {output.table}"


rule qc_table_marker_distances_calib_mini:
    input:
        imzml_peaks="{sample}/calibrated.imzML",
        mass_list="{sample}/mass_list.visualization_mini.csv",
    output:
        table="{sample}/qc/table_marker_distances_calib_mini.parquet",
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.table_marker_distances"
        " --imzml-peaks {input.imzml_peaks} --mass-list {input.mass_list}"
        " --output-table {output.table}"


rule qc_plot_marker_presence_mini:
    input:
        table_marker_distances_baseline="{sample}/qc/table_marker_distances_baseline_mini.parquet",
        table_marker_distances_calib="{sample}/qc/table_marker_distances_calib_mini.parquet",
    output:
        pdf="{sample}/qc/plot_marker_presence_mini.pdf",
    shell:
        "python -m depiction_targeted_preproc.workflow.qc.plot_marker_presence"
        " --table-marker-distances-baseline {input.table_marker_distances_baseline}"
        " --table-marker-distances-calib {input.table_marker_distances_calib}"
        " --layout-vertical"
        " --output-pdf {output.pdf}"


variants_with_map = ["mass_cluster", "reg_shift"]


rule exp_plot_map_comparison:
    input:
        mass_shifts=expand("{{sample}}/{exp_variant}/test_mass_shifts.hdf5", exp_variant=variants_with_map),
    output:
        pdf="{sample}/exp_plot_map_comparison.pdf",
    shell:
        "python -m depiction_targeted_preproc.workflow.exp.plot_map_comparison"
        " {input.mass_shifts}"
        " --output-pdf-path {output.pdf}"


# rule exp_plot_map_single_for_poster:
#    input:
#        mass_shift="{sample}/test_mass_shifts.hdf5"
#    output:
#        pdf="{sample}/exp_plot_map_single_for_poster.pdf"
#    shell:
#        "python -m depiction_targeted_preproc.workflow.exp.plot_map_single"
#        " --input-mass-shift-path {input.mass_shift}"
#        " --output-pdf-path {output.pdf}"
#
