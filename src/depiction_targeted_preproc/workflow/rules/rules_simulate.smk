# TODO this should be handled differently, i.e. the simulation code should essentially create a new
#      virtual "sample" maybe not in the raw directory just to be clear about the intentions
#   -> but then it will require some adjustment in the code that calls it

rule simulate_create_labels:
    output:
        image="{sample}/simulated_labels.hdf5",
        overview_image="{sample}/simulated_labels_overview.png",
        config="{sample}/pipeline_params.yml",
    shell:
        "python -m depiction_targeted_preproc.workflow.simulate.create_labels"
        " --output-image-path {output.image} --output-overview-image-path {output.overview_image}"
        " --config-path {output.config}"


rule simulate_create_mass_list:
    input:
        mass_list="{sample}/mass_list.calibration.csv",
        config="{sample}/pipeline_params.yml",
    output:
        mass_list="{sample}/mass_list.simulated.csv"
    shell:
        "python -m depiction_targeted_preproc.workflow.simulate.create_mass_list"
        " --input-mass-list-path {input.mass_list} --config-path {input.config}"
        " --output-mass-list-path {output.mass_list}"


rule simulate_generate_imzml:
    input:
        image="{sample}/simulated_labels.hdf5",
        mass_list="{sample}/mass_list.simulated.csv",
        config="{sample}/pipeline_params.yml",
    output:
        imzml=multiext("{sample}/simulated",".imzML",".ibd"),
    shell:
        "python -m depiction_targeted_preproc.workflow.simulate.generate_imzml"
        " --input-image-path {input.image} --input-mass-list-path {input.mass_list} --config-path {input.config}"
        " --output-imzml-path {output.imzml[0]}"