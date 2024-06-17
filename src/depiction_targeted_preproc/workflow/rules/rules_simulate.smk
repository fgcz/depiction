rule simulate_create_labels:
    output:
        image="{sample}/simulated_labels.hdf5",
        overview_image="{sample}/simulated_labels_overview.png",
        config="{sample}/pipeline_params.yml",
    shell:
        "python -m depiction_targeted_preproc.workflow.simulate.create_labels"
        " --output-image-path {output.image} --output-overview-image-path {output.overview_image}"
        " --config-path {output.config}"
