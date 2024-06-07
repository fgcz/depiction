version: "3"

include: "rules/rules_proc.smk"
include: "rules/rules_vis.smk"
include: "rules/rules_qc.smk"


exp_variants = ["chem_noise", "mass_cluster", "reg_shift"]

rule exp_compare_cluster_stats:
    input:
        csv=expand("{{sample}}/{exp_variant}/cluster_default_stats_kmeans.csv", exp_variant=exp_variants)
    output:
        pdf="{sample}/exp_compare_cluster_stats.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.exp.compare_cluster_stats"
        " {input.csv}"
        " --output-pdf {output}"

rule exp_mass_list_preparation:
    input:
        csv="{sample}/mass_list.raw.csv"
    output:
        calibration_csv="{sample}/mass_list.calibration.csv",
        standards_csv="{sample}/mass_list.standards.csv",
        visualization_csv="{sample}/mass_list.visualization.csv"
    shell:
        "python -m depiction_targeted_preproc.workflow.exp.mass_list_preparation"
        " --input-csv-path {input.csv}"
        " --out-calibration-csv-path {output.calibration_csv}"
        " --out-standards-csv-path {output.standards_csv}"
        " --out-visualization-csv-path {output.visualization_csv}"


rule exp_plot_compare_peak_density:
    input:
        tables_marker_distance=expand("{{sample}}/{exp_variant}/qc/table_marker_distances_calib.parquet", exp_variant=exp_variants),
        table_marker_distance_uncalib="{sample}/reg_shift/qc/table_marker_distances_baseline.parquet",
    output:
        pdf="{sample}/exp_plot_compare_peak_density.pdf"
    shell:
        "python -m depiction_targeted_preproc.workflow.exp.plot_compare_peak_density"
        " {input.tables_marker_distance}"
        " --table-marker-distance-uncalib {input.table_marker_distance_uncalib}"
        " --output-pdf {output.pdf}"