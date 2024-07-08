# TODO this might need to be refactored in the future, especially how binning is mixed into this
from __future__ import annotations
import functools
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from depiction.parallel_ops import ParallelConfig
from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.persistence import ImzmlModeEnum
from depiction.persistence.types import GenericReadFile, GenericReader
from depiction.spectrum.evaluate_bins import EvaluateBins


class EvaluateMeanSpectrum:
    """
    Computes mean spectra for an .imzML file.
    """

    def __init__(
        self,
        parallel_config: Optional[ParallelConfig] = None,
        eval_bins: Optional[EvaluateBins] = None,
    ) -> None:
        self._parallel_config = parallel_config
        self._eval_bins = eval_bins

    def evaluate_file(self, input_file: GenericReadFile) -> tuple[NDArray[float], NDArray[float]]:
        if input_file.imzml_mode != ImzmlModeEnum.CONTINUOUS and self._eval_bins is None:
            raise ValueError("Input file must be in 'continuous' mode.")

        # get result mz_arr
        mz_arr = self._get_result_mz_arr(input_file)

        # compute sum of spectra
        total_sum = self._get_spectra_sum(
            input_file=input_file,
            parallel_config=self._parallel_config,
            eval_bins=self._eval_bins,
        )

        # compute mean
        int_arr = total_sum / input_file.n_spectra
        return mz_arr, int_arr

    def _get_result_mz_arr(self, input_file: GenericReadFile) -> NDArray[float]:
        """Returns the m/z array for the result."""
        if self._eval_bins is None:
            with input_file.reader() as reader:
                mz_arr = reader.get_spectrum_mz(0)
        else:
            mz_arr = self._eval_bins.mz_values
        return mz_arr

    @classmethod
    def _get_spectra_sum(
        cls,
        input_file: GenericReadFile,
        parallel_config: ParallelConfig,
        eval_bins: Optional[EvaluateBins],
    ) -> NDArray[float]:
        # compute sum chunk-wise
        parallelize = ReadSpectraParallel.from_config(parallel_config)
        operation = functools.partial(cls._compute_chunk_sum, eval_bins=eval_bins)
        chunk_sums = parallelize.map_chunked(read_file=input_file, operation=operation)

        # compute sum of chunks
        return np.sum(chunk_sums, axis=0)

    @staticmethod
    def _compute_chunk_sum(
        reader: GenericReader, spectra_ids: list[int], eval_bins: Optional[EvaluateBins]
    ) -> NDArray[float]:
        if eval_bins is None:
            chunk_sum = np.array(reader.get_spectrum_int(spectra_ids[0]), copy=True)
            for i in spectra_ids[1:]:
                chunk_sum += reader.get_spectrum_int(i)
        else:
            chunk_sum = np.array(eval_bins.evaluate(*reader.get_spectrum(spectra_ids[0])), copy=True)
            for i in spectra_ids[1:]:
                chunk_sum += eval_bins.evaluate(*reader.get_spectrum(i))
        return chunk_sum
