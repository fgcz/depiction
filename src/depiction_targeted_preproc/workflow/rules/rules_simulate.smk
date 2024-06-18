# TODO this should be handled differently, i.e. the simulation code should essentially create a new
#      virtual "sample" maybe not in the raw directory just to be clear about the intentions
#   -> but then it will require some adjustment in the code that calls it

rule simulate_create_labels:
    input:
        config = "{sample}_sim/config/simulate.yml",
    output:
        image="{sample}_sim/true_labels.hdf5",
        overview_image="{sample}_sim/true_labels_overview.png",
    shell:
        "python -m depiction_targeted_preproc.workflow.simulate.create_labels"
        " --output-image-path {output.image} --output-overview-image-path {output.overview_image}"
        " --config-path {input.config}"


rule simulate_create_mass_list:
    input:
        config="{sample}_sim/config/simulate.yml",
    output:
        mass_list="{sample}_sim/mass_list.raw.csv"
    shell:
        "python -m depiction_targeted_preproc.workflow.simulate.create_mass_list"
        " --config-path {input.config}"
        " --output-mass-list-path {output.mass_list}"


rule simulate_generate_imzml:
    input:
        image="{sample}_sim/true_labels.hdf5",
        mass_list="{sample}_sim/mass_list.raw.csv",
        config="{sample}_sim/config/simulate.yml",
    output:
        imzml=multiext("{sample}_sim/raw",".imzML",".ibd"),
    shell:
        "python -m depiction_targeted_preproc.workflow.simulate.generate_imzml"
        " --input-image-path {input.image} --input-mass-list-path {input.mass_list} --config-path {input.config}"
        " --output-imzml-path {output.imzml[0]}"
