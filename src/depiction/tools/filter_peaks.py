from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlReader, ImzmlWriter
from depiction.spectrum.peak_filtering import ChainFilters, FilterNHighestIntensityPartitioned, PeakFilteringType


class FilterNHighestIntensityPartitionedConfig(BaseModel):
    method: Literal["FilterNHighestIntensityPartitioned"] = "FilterNHighestIntensityPartitioned"
    max_count: int
    n_partitions: int


class FilterPeaksConfig(BaseModel, use_enum_values=True, validate_default=True):
    filters: list[FilterNHighestIntensityPartitionedConfig]
    n_jobs: int | None = None


def _get_filter_object(config: FilterPeaksConfig) -> PeakFilteringType:
    filters = []
    for filter in config.filters:
        match filter:
            case FilterNHighestIntensityPartitionedConfig(max_count=max_count, n_partitions=n_partitions):
                filters.append(FilterNHighestIntensityPartitioned(max_count=max_count, n_partitions=n_partitions))
            case _:
                raise ValueError(f"Unknown filter method: {filter.method}")
    if len(filters) == 1:
        return filters[0]
    else:
        return ChainFilters(filters)


def _filter_chunk(
    reader: ImzmlReader, indices: list[int], writer: ImzmlWriter, peaks_filter: PeakFilteringType
) -> None:
    """Returns the filtered peaks for the given indices."""
    for spectrum_id in indices:
        mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
        mz_arr, int_arr = peaks_filter.filter_peaks(mz_arr, int_arr, mz_arr, int_arr)
        writer.add_spectrum(mz_arr, int_arr, coords)


def filter_peaks(config: FilterPeaksConfig, input_file: ImzmlReadFile, output_file: ImzmlWriteFile) -> None:
    """Filters the peaks in `input_file` and writes them to `output_file` according to the `config`."""
    peaks_filter = _get_filter_object(config)
    # TODO n_jobs handling
    parallel_config = ParallelConfig(n_jobs=config.n_jobs or 10)
    write_parallel = WriteSpectraParallel.from_config(parallel_config)
    write_parallel.map_chunked_to_file(
        read_file=input_file, write_file=output_file, operation=_filter_chunk, bind_args={"peaks_filter": peaks_filter}
    )
