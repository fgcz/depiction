from __future__ import annotations

import yaml
from functools import cached_property
from pathlib import Path
from pydantic import BaseModel, AliasPath, Field, WrapValidator
from typing import Annotated

from depiction_targeted_preproc.pipeline_config.model import (
    PipelineParameters,
    PipelineParametersPreset,
    PipelineArtifact,
)


def parse_boolean_string(v: str, handler, info) -> bool:
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise ValueError(f"Unknown boolean value: {v}")


BooleanString = Annotated[bool, WrapValidator(parse_boolean_string)]


class WorkunitConfigData(BaseModel):
    class Config:
        populate_by_name = True

    workunit_id: int = Field(validation_alias=AliasPath("job_configuration", "workunit_id"))
    project_id: int = Field(validation_alias=AliasPath("job_configuration", "project_id"))
    output_uri: str = Field(validation_alias=AliasPath("application", "output", 0))
    config_preset: str = Field(validation_alias=AliasPath("application", "parameters", "config_preset"))
    input_dataset_id: int | None = Field(
        default=None, validation_alias=AliasPath("job_configuration", "inputdataset", "_id")
    )

    output_activate_calibrated_imzml: BooleanString = Field(
        validation_alias=AliasPath("application", "parameters", "output_activate_calibrated_imzml")
    )
    output_activate_calibrated_ometiff: BooleanString = Field(
        validation_alias=AliasPath("application", "parameters", "output_activate_calibrated_ometiff")
    )
    output_activate_calibration_qc: BooleanString = Field(
        validation_alias=AliasPath("application", "parameters", "output_activate_calibration_qc")
    )

    @classmethod
    def from_yaml(cls, workunit_yaml_path: Path) -> WorkunitConfigData:
        with workunit_yaml_path.open(mode="r") as f:
            parsed = yaml.safe_load(f)
        return cls.model_validate(parsed)


class WorkunitConfig:
    """Parses the workunit configuration from Bfabric (and separates this logic from the rest of the application)."""

    def __init__(self, data: WorkunitConfigData) -> None:
        self._data = data

    @classmethod
    def from_yaml(cls, workunit_yaml_path: Path) -> WorkunitConfig:
        return cls(data=WorkunitConfigData.from_yaml(workunit_yaml_path))

    @property
    def workunit_id(self) -> int:
        return self._data.workunit_id

    @property
    def project_id(self) -> int:
        return self._data.project_id

    @property
    def input_dataset_id(self) -> int | None:
        return self._data.input_dataset_id

    @property
    def output_uri(self) -> Path:
        return Path(self._data.output_uri)

    @property
    def output_folder_absolute_path(self) -> Path:
        return Path(self._data.output_uri.split(":", 1)[1]).parent

    @cached_property
    def pipeline_parameters(self) -> PipelineParameters:
        # Load the presets
        preset_name = self._data.config_preset
        preset = PipelineParametersPreset.load_named_preset(name=preset_name)

        # Add n_jobs and requested_artifacts information to build a PipelineParameters
        return PipelineParameters.from_preset_and_settings(
            preset=preset, requested_artifacts=sorted(self.requested_artifacts), n_jobs=32
        )

    @cached_property
    def requested_artifacts(self) -> set[PipelineArtifact]:
        requested_artifacts = set()
        if self._data.output_activate_calibrated_imzml:
            requested_artifacts.add(PipelineArtifact.CALIB_IMZML)
        if self._data.output_activate_calibrated_ometiff:
            requested_artifacts.add(PipelineArtifact.CALIB_IMAGES)
        if self._data.output_activate_calibration_qc:
            requested_artifacts.add(PipelineArtifact.CALIB_QC)
        return requested_artifacts
