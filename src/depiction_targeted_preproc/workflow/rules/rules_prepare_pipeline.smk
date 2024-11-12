rule prepare_pipeline_write_pipeline_params:
    input:
        params_yml="{sample}/params.yml",
    output:
        pipeline_params_yml="{sample}/pipeline_params.yml",
    shell:
        "python -m depiction_targeted_preproc.workflow.prepare_pipeline.write_pipeline_params "
        " {input.params_yml} {output.pipeline_params_yml}"
