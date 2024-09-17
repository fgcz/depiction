from pydantic import BaseModel

from bfabric.experimental.app_interface.workunit.definition import WorkunitExecutionDefinition
from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact


class Params(BaseModel):
    config_preset: str
    requested_artifacts: list[PipelineArtifact]
    n_jobs: int = 32
    mass_list_id: int | None = None


def parse_params(definition: WorkunitExecutionDefinition) -> dict[str, str | int | bool]:
    requested_artifacts = []
    if definition.raw_parameters["output_activate_calibrated_imzml"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMZML)
    if definition.raw_parameters["output_activate_calibrated_ometiff"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMAGES)
    if definition.raw_parameters["output_activate_calibration_qc"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_QC)
    return Params(
        config_preset=definition.raw_parameters["config_preset"],
        requested_artifacts=requested_artifacts,
        mass_list_id=definition.raw_parameters.get("mass_list_id"),
    ).model_dump(mode="json")
