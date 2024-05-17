import numpy as np
from numba import njit
from numpy.typing import NDArray


class ReferencePeakDistances:
    """Methods to efficiently compute the distances between reference peaks and sample peaks."""

    @staticmethod
    @njit
    def get_distances_max_peak_in_window(
        peak_mz_arr: NDArray[float],
        peak_int_arr: NDArray[float],
        ref_mz_arr: NDArray[float],
        max_distance: float,
        max_distance_unit: str,
    ) -> NDArray[float]:
        """Returns for each reference the signed distance to the maximum peak in a window around the reference.
        The result is always returned in m/z units, regardless of the value of `max_distance_unit`.
        :param peak_mz_arr: The m/z values of the peaks.
        :param peak_int_arr: The intensities of the peaks.
        :param ref_mz_arr: The reference m/z values.
        :param max_distance: The maximum distance to consider.
        :param max_distance_unit: The unit of the distance threshold (either 'mz' or 'ppm').
        """
        n_ref = len(ref_mz_arr)
        signed_distances = np.full(n_ref, np.nan)

        for i_ref, mz_ref in enumerate(ref_mz_arr):
            if max_distance_unit == "mz":
                i_left = np.searchsorted(peak_mz_arr, mz_ref - max_distance, side="left")
                i_right = np.searchsorted(peak_mz_arr, mz_ref + max_distance, side="right")
            elif max_distance_unit == "ppm":
                mz_left = mz_ref * (1 - max_distance / 1e6)
                mz_right = mz_ref * (1 + max_distance / 1e6)
                i_left = np.searchsorted(peak_mz_arr, mz_left, side="left")
                i_right = np.searchsorted(peak_mz_arr, mz_right, side="right")
            else:
                raise ValueError(f"Unknown unit={max_distance_unit}")

            if i_left < i_right:
                i_max = i_left + np.argmax(peak_int_arr[i_left:i_right])
                s_dist_mz = peak_mz_arr[i_max] - mz_ref
            else:
                s_dist_mz = peak_mz_arr[i_left] - mz_ref

            if max_distance_unit == "mz":
                max_distance_mz = max_distance
            elif max_distance_unit == "ppm":
                max_distance_mz = max_distance / 1e6 * mz_ref

            if abs(s_dist_mz) <= max_distance_mz:
                signed_distances[i_ref] = s_dist_mz

        return signed_distances

    @staticmethod
    @njit
    def get_distances_nearest(
        peak_mz_arr: NDArray[float],
        ref_mz_arr: NDArray[float],
        max_distance_unit: str,
        max_distance: float,
    ) -> NDArray[float]:
        """Returns for each reference the signed distance to the nearest peak in the sample spectrum.
        The result is always returned in m/z units, regardless of the value of `unit`.
        :param peak_mz_arr: The m/z values of the peaks.
        :param ref_mz_arr: The reference m/z values.
        :param max_distance: The maximum distance to consider.
        :param max_distance_unit: The unit of the distance threshold (either 'mz' or 'ppm').
        """
        n_refs = len(ref_mz_arr)
        signed_distances = np.full(n_refs, np.nan)

        for i_target, mz_ref in enumerate(ref_mz_arr):
            s_dist_mz = peak_mz_arr - mz_ref
            if max_distance_unit == "mz":
                peak_index = np.argmin(np.abs(s_dist_mz))
                if np.abs(s_dist_mz[peak_index]) <= max_distance:
                    signed_distances[i_target] = s_dist_mz[peak_index]
            elif max_distance_unit == "ppm":
                dist_ppm = np.abs(s_dist_mz / mz_ref * 1e6)
                peak_index = np.argmin(dist_ppm)
                if dist_ppm[peak_index] <= max_distance:
                    signed_distances[i_target] = s_dist_mz[peak_index]
            else:
                raise ValueError(f"Unknown unit={max_distance_unit}")

        return signed_distances
