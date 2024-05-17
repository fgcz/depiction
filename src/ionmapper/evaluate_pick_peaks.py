import numpy as np
import scipy
import scipy.signal


class EvaluatePickPeaksMAD:
    """TODO this is currently a port of the approach implemented by Cardinal, but in the future it would make more sense to use a different approach."""

    def __init__(self, snr_threshold: float = 6.0, window: int = 5, blocks: int = 100):
        self._snr_threshold = snr_threshold
        self._window = window
        self._blocks = blocks

    def evaluate(self, int_array: np.ndarray[float]) -> np.ndarray[int]:
        """Returns the indices of the picked peaks."""
        # These two functions essentially implement the same algorithm as in Cardinal.
        noise = self._estimate_noise_mad(int_array, blocks=self._blocks)
        local_maxima = scipy.signal.argrelmax(int_array, order=self._window // 2)[0]
        return local_maxima[int_array[local_maxima] > self._snr_threshold * noise[local_maxima]]

    @classmethod
    def _estimate_noise_mad(cls, array, blocks=1):
        """Estimate the noise level of a spectrum using the median absolute deviation (MAD)."""
        # These two functions essentially implement the same algorithm as in Cardinal.
        blocks = max(min(blocks, len(array) // 10), 1)
        if blocks == 1:
            mad = np.median(np.abs(array - np.median(array)))
            return np.full(len(array), mad)
        else:
            t = np.arange(len(array))
            t_splits = np.array_split(t, blocks)
            x_splits = np.array_split(array, blocks)

            # For every block, compute the median absolute deviation
            x_mad = np.array([np.median(np.abs(x - np.median(x))) for x in x_splits])
            # Fit a line to the MADs
            t_mean = np.array([np.mean(t) for t in t_splits])

            # return np.interp(t, t_mean, x_mad)
            if blocks < 4:
                return np.interp(t, t_mean, x_mad)
            else:
                return scipy.interpolate.make_interp_spline(t_mean, x_mad)(t)


class EvaluatePickPeaksWellDistributed:
    """
    TODO: Urgently needs a better name
    Currently used to select reference peaks for the calibration.
    """

    def __init__(self, n_peaks_per_region: int, n_regions: int):
        self._n_peaks_per_region = n_peaks_per_region
        self._n_regions = n_regions

    def evaluate(self, mz_arr: np.ndarray, int_arr: np.ndarray) -> np.ndarray[int]:
        """Returns the indices of the selected peaks."""

        # Determine the regions
        mz_limits = np.linspace(mz_arr.min(), mz_arr.max(), self._n_regions + 1)
        mz_low = mz_limits[:-1]
        mz_high = mz_limits[1:]

        # Determine list of indices for each region
        region_indices = [np.where((low <= mz_arr) & (mz_arr < high))[0] for low, high in zip(mz_low, mz_high)]

        all_peaks = []  # type: list[int]
        for region_idx in region_indices:
            region_int = int_arr[region_idx]
            region_peaks = self._evaluate_region(region_int=region_int)
            all_peaks.extend(region_peaks)

        return np.asarray(all_peaks)

    def _evaluate_region(self, region_int: np.ndarray) -> np.ndarray[int]:
        # TODO consider a better approach here
        candidate_indices = scipy.signal.find_peaks(region_int, distance=10)[0]
        candidate_int = region_int[candidate_indices]

        # sort candidates by descending intensities
        sorted_candidate_idx = np.argsort(candidate_int)[::-1]
        return sorted_candidate_idx[: self._n_peaks_per_region]
