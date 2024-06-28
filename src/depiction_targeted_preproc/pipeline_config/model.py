from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Union, Annotated, Self

import yaml
from pydantic import BaseModel, Field, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def parse_yaml(cls, path: Path) -> Self:
        # TODO consider in the future a better mechanism for passing step configurations, maybe using
        #  json and pydantic but in a more granular way. for now this sort of works
        return cls.model_validate(yaml.unsafe_load(path.read_text()))


class BaselineAdjustmentTophat(BaseModel):
    baseline_type: Literal["Tophat"] = "Tophat"
    window_size: Union[int, float]
    window_unit: Literal["ppm", "index"]


BaselineAdjustment = Annotated[None | BaselineAdjustmentTophat, Field(discriminator="baseline_type")]


class PeakPickerBasicInterpolated(BaseModel):
    peak_picker_type: Literal["BasicInterpolated"]
    min_prominence: float
    min_distance: Union[int, float, None] = None
    min_distance_unit: Literal["index", "mz"] | None = None

    # TODO ensure min_distance are both either present or missing
    # (ideally we would just have a better typing support here and provide as tuple,
    #  but postpone for later)


class PeakPickerMSPeakPicker(BaseModel):
    peak_picker_type: Literal["MSPeakPicker"]
    fit_type: Literal["quadratic"] = "quadratic"


class PeakPickerFindMFPy(BaseModel):
    peak_picker_type: Literal["FindMFPy"]
    resolution: float = 10000.0
    width: float = 2.0
    int_width: float = 2.0
    int_threshold: float = 10.0
    area: bool = True
    max_peaks: int = 0


PeakPicker = Annotated[
    None | PeakPickerBasicInterpolated | PeakPickerMSPeakPicker | PeakPickerFindMFPy,
    Field(discriminator="peak_picker_type"),
]


class CalibrationRegressShift(BaseModel):
    calibration_method: Literal["RegressShift"]

    max_distance: float
    # TODO make explicit
    max_distance_unit: str
    # TODO make explicit
    reg_model_type: str
    # TODO make explicit
    reg_model_unit: str
    input_smoothing_activated: bool
    input_smoothing_kernel_size: int = 27
    input_smoothing_kernel_std: float = 10.0
    min_points: int = 3


class CalibrationChemicalPeptideNoise(BaseModel):
    calibration_method: Literal["ChemicalPeptideNoise"]

    n_mass_intervals: int
    interpolation_mode: Literal["linear", "cubic_spline", "refit_linear"] = "linear"
    use_ppm_space: bool = False


class CalibrationMCC(BaseModel):
    calibration_method: Literal["MCC"]

    model_smoothing_activated: bool
    model_smoothing_kernel_size: int = 27
    model_smoothing_kernel_std: float = 10.0


class CalibrationConstantGlobalShift(BaseModel):
    calibration_method: Literal["ConstantGlobalShift"]


Calibration = Annotated[
    CalibrationRegressShift | CalibrationChemicalPeptideNoise | CalibrationMCC | CalibrationConstantGlobalShift,
    Field(discriminator="calibration_method"),
]


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


# class PipelineParametersPreset(BaseModel, use_enum_values=True):


class PipelineParametersPreset(Model):
    baseline_adjustment: BaselineAdjustment
    calibration: Calibration
    peak_picker: PeakPicker
    force_peak_picker: bool


# class PipelineParameters(PipelineParametersPreset, use_enum_values=True):


class PipelineParameters(PipelineParametersPreset):
    requested_artifacts: list[PipelineArtifact]
    n_jobs: int

    @classmethod
    def from_preset_and_settings(
        cls, preset: PipelineParametersPreset, requested_artifacts: list[PipelineArtifact], n_jobs: int
    ) -> PipelineParameters:
        return cls(**preset.dict(), requested_artifacts=requested_artifacts, n_jobs=n_jobs)
