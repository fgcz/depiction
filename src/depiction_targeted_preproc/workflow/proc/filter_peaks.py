from pathlib import Path
from typing import Annotated

import typer
from typer import Option

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlReader, ImzmlWriter, ImzmlModeEnum
from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned, PeakFilteringType


def proc_filter_peaks(
    input_imzml_path: Annotated[Path, Option()], output_imzml_path: Annotated[Path, Option()]
) -> None:
    read_file = ImzmlReadFile(input_imzml_path)
    peaks_filter = FilterNHighestIntensityPartitioned(max_count=500, n_partitions=8)
    parallel_config = ParallelConfig(n_jobs=10)

    write_file = ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.PROCESSED)

    write_parallel = WriteSpectraParallel.from_config(parallel_config)
    write_parallel.map_chunked_to_file(
        read_file=read_file, write_file=write_file, operation=filter_chunk, bind_args={"peaks_filter": peaks_filter}
    )


def filter_chunk(reader: ImzmlReader, indices: list[int], writer: ImzmlWriter, peaks_filter: PeakFilteringType) -> None:
    for spectrum_id in indices:
        mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
        mz_arr, int_arr = peaks_filter.filter_peaks(mz_arr, int_arr, mz_arr, int_arr)
        writer.add_spectrum(mz_arr, int_arr, coords)


if __name__ == "__main__":
    typer.run(proc_filter_peaks)
