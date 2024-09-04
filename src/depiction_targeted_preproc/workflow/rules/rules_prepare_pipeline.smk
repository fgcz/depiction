rule prepare_pipeline_standardize_panel:
    input:
        csv="{sample}/mass_list.unstandardized.raw.csv",
    output:
        csv="{sample}/mass_list.raw.csv",
    run:
        from depiction_targeted_preproc.pipeline.setup_old import copy_standardized_table
        from pathlib import Path

        copy_standardized_table(input_csv=Path(input.csv), output_csv=Path(output.csv))


rule prepare_pipeline_write_pipeline_params:
    input:
        params_yml="params.yml",
    output:
        pipeline_params_yml="{sample}/pipeline_params.yml",
    shell:
        "python -m depiction_targeted_preproc.workflow.prepare_pipeline.write_pipeline_params "
        " {input.params_yml} {output.pipeline_params_yml}"
