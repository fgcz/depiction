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