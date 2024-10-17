from __future__ import annotations

from depiction.spectrum.peak_filtering.filter_by_snr_threshold import FilterBySnrThresholdConfig, FilterBySnrThreshold

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlReader, ImzmlWriter
from depiction.spectrum.peak_filtering import ChainFilters, FilterNHighestIntensityPartitioned, PeakFilteringType
from depiction.spectrum.peak_filtering.filter_n_highest_intensity_partitioned import (
    FilterNHighestIntensityPartitionedConfig,
)
from depiction.tools.filter_peaks.config import FilterPeaksConfig


def get_peak_filter(config: FilterPeaksConfig) -> PeakFilteringType:
    """Returns the PeakFilteringType instance as specified in config.get_peak_filtering"""
    filters = []
    for filter in config.filters:
        match filter:
            case FilterNHighestIntensityPartitionedConfig():
                filters.append(FilterNHighestIntensityPartitioned(config=filter))
            case FilterBySnrThresholdConfig():
                filters.append(FilterBySnrThreshold(config=filter))
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
    peaks_filter = get_peak_filter(config)
    # TODO n_jobs handling
    parallel_config = ParallelConfig(n_jobs=config.n_jobs or 10)
    write_parallel = WriteSpectraParallel.from_config(parallel_config)
    write_parallel.map_chunked_to_file(
        read_file=input_file, write_file=output_file, operation=_filter_chunk, bind_args={"peaks_filter": peaks_filter}
    )
