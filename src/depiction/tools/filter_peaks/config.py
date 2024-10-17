from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, PositiveInt

from depiction.spectrum.peak_filtering.filter_by_snr_threshold import FilterBySnrThresholdConfig


class FilterNHighestIntensityPartitionedConfig(BaseModel):
    method: Literal["FilterNHighestIntensityPartitioned"] = "FilterNHighestIntensityPartitioned"
    max_count: int
    n_partitions: int


class FilterPeaksConfig(BaseModel, use_enum_values=True, validate_default=True):
    filters: list[FilterNHighestIntensityPartitionedConfig | FilterBySnrThresholdConfig]
    n_jobs: PositiveInt | None = None
