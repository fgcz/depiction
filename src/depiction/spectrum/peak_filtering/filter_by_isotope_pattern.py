from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from depiction.calibration.isotope_pattern_matcher import IsotopePatternMatcher


@dataclass
class FilterByIsotopePattern:
    """Filters peaks by comparing their isotopic pattern to the expected averagine model isotopic pattern.
    :param agreement_threshold: A minimum threshold on the cosine similarity between the observed and expected isotopic
        peak intensities.
    :param n_isotopic_peaks_min: The minimum number of isotopic peaks which must be identified at a particular mass,
        in order for the peak to be retained.
    :param n_isotopic_peaks_max: The maximum number of isotopic peaks which will be considered for a particular mass,
        further peaks are allowed but will not be considered for the agreement score.
    :param mass_distance_tolerance: The maximum distance between the observed and expected isotopic peak masses, i.e.
        in terms of mz values which will first be matched to the expected isotopic peak masses.
        In general the first mass will always be defined such that the distance is zero.
    """

    agreement_threshold: float
    n_isotopic_peaks_min: int
    n_isotopic_peaks_max: int
    mass_distance_tolerance: float

    def __post_init__(self):
        self._isotope_pattern_matcher = IsotopePatternMatcher(
            cache_size=1000,
            cache_tolerance=0.5,
        )

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_idx_arr: NDArray[int],
    ) -> NDArray[int]:
        """Returns the subset of the peak indices, filtered such that each group of isotopic peaks matches the expected
        averagine model isotopic pattern at the given mass."""
        if len(peak_idx_arr) == 0:
            return np.array([], dtype=int)

        (
            agreement_scores,
            agreement_lengths,
        ) = self._isotope_pattern_matcher.compute_averagine_agreement_at_positions(
            mz_peaks=spectrum_mz_arr[peak_idx_arr],
            int_peaks=spectrum_int_arr[peak_idx_arr],
            idx_positions=np.arange(len(peak_idx_arr)),
            n_limit=self.n_isotopic_peaks_max,
            distance_tolerance=self.mass_distance_tolerance,
        )
        agreement_ok = np.zeros(len(agreement_scores), dtype=bool)
        for i in range(len(agreement_scores)):
            if agreement_scores[i] >= self.agreement_threshold and agreement_lengths[i] >= self.n_isotopic_peaks_min:
                agreement_ok[i : i + agreement_lengths[i]] = True
        return peak_idx_arr[agreement_ok]
