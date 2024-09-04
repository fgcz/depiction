from pathlib import Path

import yaml
from bfabric import Bfabric
from bfabric.entities import Workunit
from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact
from pydantic import BaseModel


class Params(BaseModel):
    config_preset: str
    requested_artifacts: list[PipelineArtifact]
    n_jobs: int = 32


def _get_params(client: Bfabric, workunit_id: int) -> dict[str, str | int | bool]:
    workunit = Workunit.find(id=workunit_id, client=client)
    requested_artifacts = []
    if workunit.parameter_values["output_activate_calibrated_imzml"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMZML)
    if workunit.parameter_values["output_activate_calibrated_ometiff"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMAGES)
    if workunit.parameter_values["output_activate_calibration_qc"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_QC)
    return Params(
        config_preset=workunit.parameter_values["config_preset"], requested_artifacts=requested_artifacts
    ).model_dump(mode="json")


def prepare_params(
    client: Bfabric,
    sample_dir: Path,
    workunit_id: int,
) -> None:
    params_yaml = sample_dir / "params.yml"
    with params_yaml.open("w") as file:
        yaml.safe_dump(_get_params(client=client, workunit_id=workunit_id), file)
