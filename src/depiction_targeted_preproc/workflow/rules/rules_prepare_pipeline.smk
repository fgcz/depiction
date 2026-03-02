rule prepare_pipeline_write_pipeline_params:
    input:
        params_yml="{sample}/params.yml",
    output:
        pipeline_params_yml="{sample}/pipeline_params.yml",
    shell:
        "python -m depiction_targeted_preproc.workflow.prepare_pipeline.write_pipeline_params "
        " {input.params_yml} {output.pipeline_params_yml}"


# Workaround to support .imzml.zip without changing the whole workflow.
rule prepare_pipeline_extract_imzml_zip:
    input:
        zip="{sample}/raw.imzML.zip",
    output:
        imzml="{sample}/raw.imzML",
        ibd="{sample}/raw.ibd",
    run:
        import zipfile

        with zipfile.ZipFile(input.zip, "r") as archive:
            filenames = archive.namelist()
            imzml_filename = next(filename for filename in filenames if filename.lower().endswith(".imzml"))
            ibd_filename = next(filename for filename in filenames if filename.lower().endswith(".ibd"))
            archive.extract(imzml_filename, path=output.imzml)
            archive.extract(ibd_filename, path=output.ibd)
