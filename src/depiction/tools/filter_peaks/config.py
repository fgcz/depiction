from __future__ import annotations

from pydantic import BaseModel, PositiveInt

from depiction.spectrum.peak_filtering.filter_by_snr_threshold import FilterBySnrThresholdConfig
from depiction.spectrum.peak_filtering.filter_n_highest_intensity_partitioned import (
    FilterNHighestIntensityPartitionedConfig,
)


class FilterPeaksConfig(BaseModel, use_enum_values=True, validate_default=True):
    filters: list[FilterNHighestIntensityPartitionedConfig | FilterBySnrThresholdConfig]
    n_jobs: PositiveInt | None = None
