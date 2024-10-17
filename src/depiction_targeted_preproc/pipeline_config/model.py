from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict

from depiction.tools.calibrate.config import CalibrationConfig
from depiction.tools.correct_baseline.config import BaselineCorrectionConfig
from depiction.tools.filter_peaks.config import FilterPeaksConfig
from depiction.tools.pick_peaks import PickPeaksConfig


class Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def parse_yaml(cls, path: Path) -> Self:
        # TODO consider in the future a better mechanism for passing step configurations, maybe using
        #  json and pydantic but in a more granular way. for now this sort of works
        return cls.model_validate(yaml.unsafe_load(path.read_text()))


class SimulateParameters(Model):
    image_width: int = 200
    image_height: int = 100
    n_labels: int = 30
    bin_width_ppm: float = 100.0
    target_mass_min: float = 850.0
    target_mass_max: float = 1900.0


class PipelineArtifact(str, Enum):
    CALIB_IMZML = "CALIB_IMZML"
    CALIB_IMAGES = "CALIB_IMAGES"
    CALIB_QC = "CALIB_QC"
    CALIB_HEATMAP = "CALIB_HEATMAP"
    DEBUG = "DEBUG"


# TODO we have some redundancy in the peak picker allowing filtering... it should be removed there, since sometimes you
#      might want peak filtering without peak picking -> TODO create task
class PipelineParametersPreset(Model, use_enum_values=True, validate_default=True):
    model_config = ConfigDict(extra="forbid")

    baseline_correction: BaselineCorrectionConfig | None
    filter_peaks: FilterPeaksConfig | None
    calibration: CalibrationConfig | None
    pick_peaks: PickPeaksConfig | None

    @classmethod
    def get_preset_path(cls, name: str) -> Path:
        """Returns the path to the preset file with the specified name."""
        return Path(__file__).parents[1] / "app" / "config_presets" / f"{name}.yml"

    @classmethod
    def load_named_preset(cls, name: str) -> PipelineParametersPreset:
        """Loads named presets distributed with depiction_targeted_preproc."""
        return cls.parse_yaml(cls.get_preset_path(name))


class PipelineParameters(PipelineParametersPreset, use_enum_values=True, validate_default=True):
    requested_artifacts: list[PipelineArtifact]
    n_jobs: int

    @classmethod
    def from_preset_and_settings(
        cls, preset: PipelineParametersPreset, requested_artifacts: list[PipelineArtifact], n_jobs: int
    ) -> PipelineParameters:
        return cls(**preset.model_dump(), requested_artifacts=requested_artifacts, n_jobs=n_jobs)
