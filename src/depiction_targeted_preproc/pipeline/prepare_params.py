from typing import Any

from pydantic import BaseModel

from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact


class Params(BaseModel):
    config_preset: str
    requested_artifacts: list[PipelineArtifact]
    # TODO handle this default value better as it can cause issues often
    n_jobs: int = 10
    mass_list_id: int | None = None


def parse_params(raw_parameters: dict[str, Any]) -> dict[str, str | int | bool]:
    requested_artifacts = []
    if raw_parameters["output_activate_calibrated_imzml"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMZML)
    if raw_parameters["output_activate_calibrated_ometiff"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMAGES)
    if raw_parameters["output_activate_calibration_qc"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_QC)
    return Params(
        config_preset=raw_parameters["config_preset"],
        requested_artifacts=requested_artifacts,
        mass_list_id=raw_parameters.get("mass_list_id"),
    ).model_dump(mode="json")
