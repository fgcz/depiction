import argparse

import numpy as np
from numpy.typing import NDArray

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import (
    ImzmlReadFile,
    ImzmlWriteFile,
    ImzmlReader,
    ImzmlWriter,
)


class LimitMzRange:
    def __init__(self, mz_range: tuple[float, float]) -> None:
        self._mz_range = mz_range

    def evaluate_spectrum(
        self, mz_arr: NDArray[float], int_arr: NDArray[float]
    ) -> tuple[NDArray[float], NDArray[float]]:
        return self._evaluate_spectrum(mz_arr, int_arr, self._mz_range)

    def evaluate_file(
        self,
        read_file: ImzmlReadFile,
        write_file: ImzmlWriteFile,
        parallel_config: ParallelConfig,
    ) -> None:
        def chunk_operation(reader: ImzmlReader, spectra_ids: list[int], writer: ImzmlWriter, mz_range) -> None:
            for spectrum_id in spectra_ids:
                mz_arr, int_arr = reader.get_spectrum(spectrum_id)
                mz_arr_new, int_arr_new = self._evaluate_spectrum(mz_arr, int_arr, mz_range)
                writer.add_spectrum(mz_arr_new, int_arr_new, reader.coordinates[spectrum_id])

        parallelize = WriteSpectraParallel.from_config(config=parallel_config)
        parallelize.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=chunk_operation,
            bind_args={"mz_range": self._mz_range},
        )

    @staticmethod
    def _evaluate_spectrum(
        mz_arr: NDArray[float], int_arr: NDArray[float], mz_range: tuple[float, float]
    ) -> tuple[NDArray[float], NDArray[float]]:
        left_index = np.searchsorted(mz_arr, mz_range[0], side="left")
        right_index = np.searchsorted(mz_arr, mz_range[1], side="right")
        return mz_arr[left_index:right_index], int_arr[left_index:right_index]


def main_limit_mz_range(input_file: str, output_file: str, mz_range: tuple[float, float], n_jobs: int) -> None:
    # check the input file
    print("Checking the input file:")
    read_file = ImzmlReadFile(input_file)
    read_file.print_summary()
    # set up for output
    write_file = ImzmlWriteFile(output_file, imzml_mode=read_file.imzml_mode)
    parallel_config = ParallelConfig(n_jobs=n_jobs)
    # perform the operation
    print("Starting file transformation")
    limit = LimitMzRange(mz_range=mz_range)
    limit.evaluate_file(read_file=read_file, write_file=write_file, parallel_config=parallel_config)
    # finally check the output
    print("Output information:")
    ImzmlReadFile(output_file).print_summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="The input imzML file to read.")
    parser.add_argument("--output-file", type=str, help="The output imzML file to write.")
    parser.add_argument(
        "--mz-range",
        nargs=2,
        type=float,
        help="The mz range to limit the spectra to, e.g. 100 200.",
    )
    parser.add_argument("--n-jobs", type=int, default=20, help="The number of parallel jobs to use.")

    args = vars(parser.parse_args())
    main_limit_mz_range(**args)


if __name__ == "__main__":
    main()
