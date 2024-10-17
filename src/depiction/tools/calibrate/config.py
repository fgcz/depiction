from __future__ import annotations

from typing import Literal, Annotated

from pydantic import BaseModel, Field

from depiction.tools.filter_peaks.config import FilterPeaksConfig


class CalibrationRegressShiftConfig(BaseModel):
    calibration_method: Literal["RegressShift"] = "RegressShift"

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

    peak_filtering: FilterPeaksConfig | None = None


class CalibrationChemicalPeptideNoiseConfig(BaseModel):
    calibration_method: Literal["ChemicalPeptideNoise"] = "ChemicalPeptideNoise"

    n_mass_intervals: int
    interpolation_mode: Literal["linear", "cubic_spline", "refit_linear"] = "linear"
    use_ppm_space: bool = False


class CalibrationMCCConfig(BaseModel):
    calibration_method: Literal["MCC"] = "MCC"

    coef_smoothing_activated: bool
    coef_smoothing_kernel_size: int = 27
    coef_smoothing_kernel_std: float = 10.0


class CalibrationConstantGlobalShiftConfig(BaseModel):
    calibration_method: Literal["ConstantGlobalShift"] = "ConstantGlobalShift"


class CalibrationConfig(BaseModel, use_enum_values=True, validate_default=True):
    method: (
        Annotated[
            CalibrationRegressShiftConfig
            | CalibrationChemicalPeptideNoiseConfig
            | CalibrationMCCConfig
            | CalibrationConstantGlobalShiftConfig,
            Field(discriminator="calibration_method"),
        ]
        | None
    )

    n_jobs: int = 1
