from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import ImzmlModeEnum
from depiction.persistence.types import GenericWriteFile, GenericReadFile, GenericWriter, GenericReader


@dataclass
class ResampleMassAxis:
    target_mz_arr: NDArray[float]

    def evaluate_spectrum(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        """Resamples the given spectrum to the target mass axis."""
        spline = CubicSpline(x=mz_arr, y=int_arr, extrapolate=False)
        values = spline(self.target_mz_arr)
        return np.nan_to_num(values, nan=0)

    def evaluate_file(
        self,
        read_file: GenericReadFile,
        write_file: GenericWriteFile,
        parallel_config: ParallelConfig,
        allow_processed: bool = False,
    ) -> None:
        if not allow_processed and write_file.imzml_mode != ImzmlModeEnum.CONTINUOUS:
            raise ValueError("Interpolation is only supported for continuous imzML/profile spectra.")

        write_parallel = WriteSpectraParallel.from_config(config=parallel_config)
        write_parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._evaluate_file_chunk,
            bind_args={
                "target_mz_arr": self.target_mz_arr,
            },
        )

    @classmethod
    def _evaluate_file_chunk(
        cls, reader: GenericReader, spectra_ids: list[int], writer: GenericWriter, target_mz_arr: NDArray[float]
    ) -> None:
        resampler = ResampleMassAxis(target_mz_arr=target_mz_arr)

        for spectrum_id in spectra_ids:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            int_arr = resampler.evaluate_spectrum(mz_arr, int_arr)
            writer.add_spectrum(mz_arr, int_arr, coordinates=coords)
