from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReferenceDistanceEstimator:
    """
    Estimates the distances between peaks in a sample spectrum and a reference spectrum.
    For every reference peak, the nearest sample spectrum peak is identified and the distance to it's immediate neighbours is computed.
    The number of neighbours is determined by the parameter n_candidates, and has to be odd so the nearest peak's distance is in the middle.
    :param reference_mz: m/z values of the reference spectrum
    :param n_candidates: number of candidates to consider for each peak to consider
    """

    reference_mz: NDArray[float]
    n_candidates: int

    def __post_init__(self):
        self.reference_mz = np.asarray(self.reference_mz)
        if self.n_candidates % 2 == 0:
            raise ValueError("n_candidates must be odd")

    @property
    def n_targets(self) -> int:
        """Returns the number of reference peaks."""
        return len(self.reference_mz)

    @property
    def closest_index(self) -> int:
        """Returns the index of the closest peak."""
        return self.n_candidates // 2

    # TODO not fully sure if this is the correct place
    # TODO also, it's not clear whether it would be better/possible to use a strategy that also takes into account outliers already here
    def compute_max_peak_within_distance(
        self,
        mz_peaks: NDArray[float],
        int_peaks: NDArray[float],
        max_distance: float,
        keep_missing: bool = False,
    ) -> tuple[NDArray[int], NDArray[float]]:
        """
        For every reference peak, a nearby peak in ``mz_peaks`` is identified by finding the peak with the highest intensity.
        If no peak is found within ``max_distance``, the entry is skipped, i.e. the list might have less entries than ``n_targets``.
        :param mz_peaks: m/z values of the sample spectrum's peaks
        :param int_peaks: intensities of the sample spectrum's peaks
        :param max_distance: maximum distance to consider
        :param keep_missing: if True, missing peaks are indicated by -1 in the indices and NaN in the distances
        :return:
        - indices: a numpy array containing the indices in mz_peaks/int_peaks of the chosen peaks
        - signed_distances: a numpy array containing the signed distances to the reference peaks, i.e. the error
        """
        if len(mz_peaks) == 0:
            raise ValueError("mz_peaks must not be empty")

        indices = []
        signed_distances = []
        for mz_reference in self.reference_mz:
            abs_distances = np.abs(mz_peaks - mz_reference)
            idx_valid = np.where(abs_distances <= max_distance)[0]
            if not idx_valid.size:
                # TODO make keep_missing the default
                if keep_missing:
                    indices.append(-1)
                    signed_distances.append(np.nan)
                continue
            idx_max = idx_valid[np.argmax(int_peaks[idx_valid])]
            indices.append(idx_max)
            # "value +0.2 means that the peak is 0.2 m/z higher than the reference"
            signed_distances.append(mz_peaks[idx_max] - mz_reference)

        return np.asarray(indices), np.asarray(signed_distances)

    def compute_distances_for_peaks(self, mz_peaks: NDArray[float]) -> tuple[NDArray[float], NDArray[int]]:
        """
        For every reference peak, the closest correspondence in ``mz_peaks`` and its neighbours are identified
        and their (signed) distances
        $$mz_\\text{peaks} - mz_\\text{reference}$$
        will be returned.
        The method also returns the indices of the closest peaks, whose distance is found at ``distances[i, closest_indices[i]]``.
        :param mz_peaks: m/z values of the sample spectrum's peaks
        :return:
        - distances: a matrix of shape (n_targets, n_candidates) where each row corresponds to a target and each column to a candidate,
            the values indicate the signed distance to the target
        - closest_indices: a vector of length n_targets containing the indices of the nearest candidates (in the
            mz_peaks array) for each target (i.e. the index of the center element of the corresponding row in distances)
        """
        if len(mz_peaks) == 0:
            raise ValueError("mz_peaks must not be empty")

        nearest_indices = np.zeros(self.n_targets, dtype=int)
        distances = np.zeros((self.n_targets, self.n_candidates))

        for i_target, mz_reference in enumerate(self.reference_mz):
            # "value +0.2 means that the peak is 0.2 m/z higher than the reference"
            signed_distances = mz_peaks - mz_reference

            # find the closest value
            index_min = np.argmin(np.abs(signed_distances))
            nearest_indices[i_target] = index_min

            # now, computing the neighbourhood distances is easy, but it requires careful consideration of corner cases
            distances[i_target] = self._convert_to_row(signed_distances, index_min)

        return distances, nearest_indices

    def _convert_to_row(self, signed_distances: NDArray[float], index_min: int) -> NDArray[float]:
        """
        Extracts the neighborhood of the closest peak from the signed distances.
        This method also handles corner cases at the beginning and the end of the peak list,
        in which case distance is set to -inf or +inf, respectively.
        :param signed_distances: a vector of signed distances
        :param index_min: the index of the closest peak
        """
        half_length = self.n_candidates // 2
        i_left = index_min - half_length
        i_right = index_min + half_length + 1

        # NOTE: The code assumes that it's not possible to be out-of-bounds on both sides simultaneously
        if i_left < 0:
            # pad with -inf from the left
            return np.array([*np.repeat([-np.inf], -i_left), *signed_distances[:i_right]])
        elif i_right > len(signed_distances):
            # pad with +inf from the right
            return np.array(
                [
                    *signed_distances[i_left:],
                    *np.repeat([np.inf], i_right - len(signed_distances)),
                ]
            )
        else:
            return signed_distances[i_left:i_right]
