import argparse
import dataclasses

import numba
import numpy as np

from ionmapper.parallel_ops import ParallelConfig, WriteSpectraParallel
from ionmapper.persistence import ImzmlWriteFile, ImzmlReadFile, ImzmlWriter, ImzmlReader


class NormalizeSpectraIntensitiesVariant:
    TIC: str = "TIC"
    MEDIAN: str = "MEDIAN"
    VEC_NORM: str = "VEC_NORM"


@dataclasses.dataclass
class NormalizeSpectraIntensities:
    variant: NormalizeSpectraIntensitiesVariant

    def process_file(self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile, parallel_config: ParallelConfig):
        write_parallel = WriteSpectraParallel.from_config(parallel_config)
        write_parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._process_chunk,
            bind_args={"variant": self.variant},
        )

    @classmethod
    def _process_chunk(cls, reader: ImzmlReader, indices: list[int], writer: ImzmlWriter, variant: str):
        method = cls._get_operation(variant=variant)
        for id_spectrum in indices:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(id_spectrum)
            int_arr = method(int_arr)
            writer.add_spectrum(mz_arr, int_arr, coords)

    @classmethod
    def _get_operation(cls, variant: str):
        if variant == NormalizeSpectraIntensitiesVariant.TIC:
            return _normalize_tic
        elif variant == NormalizeSpectraIntensitiesVariant.MEDIAN:
            return _normalize_median
        elif variant == NormalizeSpectraIntensitiesVariant.VEC_NORM:
            return _normalize_vec_norm
        else:
            raise ValueError(f"Unknown variant: {variant}")


@numba.njit("float64[:](float64[:])")
def _normalize_tic(int_arr):
    tic = int_arr.sum()
    return int_arr / tic


@numba.njit("float64[:](float64[:])")
def _normalize_median(int_arr):
    return int_arr / np.median(int_arr)


@numba.njit("float64[:](float64[:])")
def _normalize_vec_norm(int_arr):
    return int_arr / np.linalg.norm(int_arr)


def main_normalize_intensities(
    input_imzml: str, output_imzml: str, variant: NormalizeSpectraIntensitiesVariant, n_jobs: int
):
    parallel_config = ParallelConfig(n_jobs=n_jobs, task_size=None)
    with ImzmlReadFile(input_imzml) as read_file:
        with ImzmlWriteFile(output_imzml, imzml_mode=read_file.imzml_mode) as write_file:
            normalize_intensities = NormalizeSpectraIntensities(variant=variant)
            normalize_intensities.process_file(
                read_file=read_file, write_file=write_file, parallel_config=parallel_config
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-imzml", required=True)
    parser.add_argument("--output-imzml", required=True)
    parser.add_argument("--variant", required=True, choices=["tic", "median", "vec_norm"])
    parser.add_argument("--n-jobs", type=int, default=20)
    args = vars(parser.parse_args())
    args["variant"] = NormalizeSpectraIntensitiesVariant(args["variant"].upper())
    main_normalize_intensities(**args)


if __name__ == "__main__":
    main()
