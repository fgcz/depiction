from pathlib import Path

import cyclopts
import yaml
from bfabric import Bfabric
from bfabric.entities import Workunit
from bfabric.experimental.app_interface.workunit.definition import WorkunitExecutionDefinition
from loguru import logger
from pydantic import BaseModel

from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact


class Params(BaseModel):
    config_preset: str
    requested_artifacts: list[PipelineArtifact]
    n_jobs: int = 32


def parse_params(definition: WorkunitExecutionDefinition) -> dict[str, str | int | bool]:
    requested_artifacts = []
    if definition.raw_parameters["output_activate_calibrated_imzml"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMZML)
    if definition.raw_parameters["output_activate_calibrated_ometiff"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMAGES)
    if definition.raw_parameters["output_activate_calibration_qc"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_QC)
    return Params(
        config_preset=definition.raw_parameters["config_preset"], requested_artifacts=requested_artifacts
    ).model_dump(mode="json")


def prepare_params(
    client: Bfabric,
    sample_dir: Path,
    workunit_id: int,
    override: bool,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    params_yaml = sample_dir / "params.yml"
    if params_yaml.is_file() and not override:
        logger.info(f"Skipping params generation for {workunit_id} as it already exists and override is not set")
        return
    definition = WorkunitExecutionDefinition.from_workunit(Workunit.find(id=workunit_id, client=client))
    with params_yaml.open("w") as file:
        yaml.safe_dump(parse_params(definition), file)


app = cyclopts.App()


@app.default
def prepare_params_from_cli(
    sample_dir: Path,
    config_preset: str,
    requested_artifacts: list[str],
    n_jobs: int = 32,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    params_yaml = sample_dir / "params.yml"
    with params_yaml.open("w") as file:
        yaml.safe_dump(
            Params(config_preset=config_preset, requested_artifacts=requested_artifacts, n_jobs=n_jobs).model_dump(
                mode="json"
            ),
            file,
        )


if __name__ == "__main__":
    app()
