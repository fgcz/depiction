import enum

import numba
import numpy as np
from numpy.typing import NDArray

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlReader, ImzmlWriter


class BinStatistic(enum.Enum):
    """
    The statistic to use for evaluating the bins.
    """

    SUM = 0
    MEAN = 1


class EvaluateBins:
    """
    Evaluates the binning of spectra, for specific m/z values.
    """

    def __init__(self, bin_edges: NDArray[float], statistic: BinStatistic = BinStatistic.MEAN) -> None:
        """
        :param bin_edges: The bin edges to use for binning the spectra, including the lower and upper bounds.
        :param statistic: The statistic to use for evaluating the bins.
        """
        self._bin_edges = np.asarray(bin_edges)
        self._statistic = statistic

    def evaluate(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        """
        Evaluates the binning for the provided spectrum of m/z and intensity values.
        :param mz_arr: The m/z values of the spectrum.
        :param int_arr: The intensity values of the spectrum.
        :return: The binned intensities.
        """
        is_f64 = isinstance(mz_arr.dtype, (float, np.float64)) or isinstance(int_arr.dtype, (float, np.float64))
        dtype = np.float64 if is_f64 else np.float32
        return self._compute_evaluate(
            mz_arr.astype(dtype),
            int_arr.astype(dtype),
            self._bin_edges.astype(dtype),
            self._statistic.value,
        )

    def evaluate_file(
        self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile, parallel_config: ParallelConfig
    ) -> None:
        write_parallel = WriteSpectraParallel.from_config(parallel_config)
        write_parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._compute_chunk,
            bind_args={
                "bin_edges": self._bin_edges,
                "statistic": self._statistic.value,
            },
        )

    @staticmethod
    def _compute_chunk(
        reader: ImzmlReader,
        spectra_ids: list[int],
        writer: ImzmlWriter,
        bin_edges: NDArray[float],
        statistic: int,
    ) -> None:
        """
        Computes the chunk of spectra for the provided reader and writer, and the given bin edges.
        """
        eval_bins = EvaluateBins(bin_edges=bin_edges, statistic=BinStatistic(statistic))
        for spectrum_id in spectra_ids:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            binned_int_arr = eval_bins.evaluate(mz_arr, int_arr)
            writer.add_spectrum(eval_bins.mz_values, binned_int_arr, coords)

    @staticmethod
    @numba.njit(
        [
            numba.float32[:](numba.float32[:], numba.float32[:], numba.float32[:], numba.int32),
            numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:], numba.int32),
        ],
        error_model="numpy",
    )
    def _compute_evaluate(
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
        bin_edges: NDArray[float],
        statistic: int,
    ) -> NDArray[float]:
        """
        Evaluates the binning for the provided spectrum of m/z and intensity values, accelerated with numba.
        """
        # In principle, this could be implemented more nicely with digitize, in practice it is slower that way.
        n_bins = bin_edges.shape[0] - 1
        bin_array = np.zeros(n_bins, dtype=int_arr.dtype)

        mz_arr = np.sort(mz_arr)
        lower_bound_indices = np.searchsorted(mz_arr, bin_edges[:-1], side="left")
        upper_bound_indices = np.searchsorted(mz_arr, bin_edges[1:], side="right")

        if statistic == 0:
            for i_bin in range(n_bins):
                value = np.sum(int_arr[lower_bound_indices[i_bin] : upper_bound_indices[i_bin]])
                bin_array[i_bin] = value if np.isfinite(value) else 0
        elif statistic == 1:
            for i_bin in range(n_bins):
                value = np.mean(int_arr[lower_bound_indices[i_bin] : upper_bound_indices[i_bin]])
                bin_array[i_bin] = value if np.isfinite(value) else 0
        else:
            raise RuntimeError(f"Unsupported statistic {statistic}.")

        return bin_array

    @property
    def mz_values(self) -> NDArray[float]:
        """The m/z values of this binning, corresponding to the centers of each bin."""
        return (self._bin_edges[:-1] + self._bin_edges[1:]) / 2

    @property
    def bin_edges(self) -> NDArray[float]:
        """The bin edges."""
        view = self._bin_edges.view()
        view.flags.writeable = False
        return view

    @classmethod
    def from_mz_values(cls, mz_values: NDArray[float]) -> "EvaluateBins":
        """Constructs an instance of EvaluateBins from the provided m/z values, i.e. at the center of bins."""
        bin_edges = np.zeros(mz_values.shape[0] + 1, dtype=mz_values.dtype)
        bin_edges[1:-1] = (mz_values[1:] + mz_values[:-1]) / 2

        # The two outermost edges assume a symmetric bin.
        bin_edges[0] = mz_values[0] - (mz_values[1] - mz_values[0]) / 2
        bin_edges[-1] = mz_values[-1] + (mz_values[-1] - mz_values[-2]) / 2

        return cls(bin_edges=bin_edges)
