from __future__ import annotations

from typing import Literal, Self, Annotated

from pydantic import BaseModel, model_validator, Field

from depiction.tools.filter_peaks.config import FilterPeaksConfig


class PeakPickerBasicInterpolatedConfig(BaseModel):
    peak_picker_type: Literal["BasicInterpolated"] = "BasicInterpolated"
    min_prominence: float
    min_distance: int | float | None = None
    min_distance_unit: Literal["index", "mz"] | None = None

    @model_validator(mode="after")
    def validate_min_distance(self) -> Self:
        if self.min_distance is not None and self.min_distance_unit is None:
            raise ValueError("min_distance_unit must be provided if min_distance is set")
        if self.min_distance_unit is not None and self.min_distance is None:
            raise ValueError("min_distance must be provided if min_distance_unit is set")
        return self


class PeakPickerBasicUninterpolatedConfig(BaseModel):
    peak_picker_type: Literal["BasicUninterpolated"] = "BasicUninterpolated"
    min_prominence: float
    min_distance: int | float | None = None
    min_distance_unit: Literal["index", "mz"] | None = None
    # TODO make optional later
    smooth_sigma: Annotated[float, Field(gt=0)] = 0.0

    @model_validator(mode="after")
    def validate_min_distance(self) -> Self:
        if self.min_distance is not None and self.min_distance_unit is None:
            raise ValueError("min_distance_unit must be provided if min_distance is set")
        if self.min_distance_unit is not None and self.min_distance is None:
            raise ValueError("min_distance must be provided if min_distance_unit is set")
        return self


class PeakPickerMSPeakPickerConfig(BaseModel):
    peak_picker_type: Literal["MSPeakPicker"] = "MSPeakPicker"
    fit_type: Literal["quadratic"] = "quadratic"


class PeakPickerFindMFPyConfig(BaseModel):
    peak_picker_type: Literal["FindMFPy"] = "FindMFPy"
    resolution: float = 10000.0
    width: float = 2.0
    int_width: float = 2.0
    int_threshold: float = 10.0
    area: bool = True
    max_peaks: int = 0


class PickPeaksConfig(BaseModel, use_enum_values=True, validate_default=True):
    peak_picker: (
        PeakPickerBasicInterpolatedConfig
        | PeakPickerBasicUninterpolatedConfig
        | PeakPickerMSPeakPickerConfig
        | PeakPickerFindMFPyConfig
    ) = Field(..., discriminator="peak_picker_type")
    n_jobs: int
    force_peak_picker: bool = False
    peak_filtering: FilterPeaksConfig | None = None
