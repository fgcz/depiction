"""This file bundles all panel manipulation rules needed throughout the pipeline.
To keep things tidy, each file will be in the same directory `{sample}/panels/`.
Initially before the workflow is started a file `mass_list.raw.csv` has to be created.
"""


rule panel_standardized_full:
    input:
        csv="{sample}/panels/unstandardized_full.csv",
    output:
        csv="{sample}/panels/full.csv",
    shell:
        "python -m depiction_targeted_preproc.workflow.panel.standardize_panel "
        "--input-panel-path {input.csv} --config-name standardize_main_config.yml "
        "--output-panel-path {output.csv}"


# TODO use this in the visualization steps everywhere
rule panel_visualize_full:
    input:
        csv="{sample}/panels/full.csv",
    output:
        csv="{sample}/panels/full_visualize.csv",
    run:
        import polars as pl

        # TODO this absolutely needs to be configurable too
        df = pl.read_csv(input.csv).with_columns(tol=pl.lit(0.25)).drop("type")
        df.write_csv(output.csv)


rule panel_filter_calibration:
    input:
        csv="{sample}/panels/full.csv",
    output:
        csv="{sample}/panels/calibration.csv",
    run:
        import polars as pl

        pl.read_csv(input.csv).filter(type="target").write_csv(output.csv)
