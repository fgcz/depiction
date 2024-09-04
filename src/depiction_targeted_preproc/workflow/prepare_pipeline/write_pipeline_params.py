from pathlib import Path

import cyclopts
import yaml
from depiction_targeted_preproc.pipeline.prepare_params import Params
from depiction_targeted_preproc.pipeline_config.model import PipelineParametersPreset, PipelineParameters
from loguru import logger

app = cyclopts.App()


@app.default()
def write_pipeline_params(
    input_params_yml_path: Path,
    output_pipeline_params_yml_path: Path,
):
    params = Params.model_validate(yaml.safe_load(input_params_yml_path.read_text()))
    logger.info(f"Preparing pipeline parameters for preset {params.preset_name}")

    preset = PipelineParametersPreset.load_named_preset(name=params.preset_name)
    # Add n_jobs and requested_artifacts information to build a PipelineParameters
    pipeline_params = PipelineParameters.from_preset_and_settings(
        preset=preset, requested_artifacts=params.requested_artifacts, n_jobs=params.n_jobs
    )
    # Write to the file
    with output_pipeline_params_yml_path.open("w") as file:
        yaml.dump(pipeline_params.model_dump(mode="json"), file)


if __name__ == "__main__":
    app()
