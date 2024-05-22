from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import alphapept
import alphapept.chem
import alphapept.constants
import numpy as np
from numpy.typing import NDArray

from depiction.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker
from depiction.misc.numpy_util import NumpyUtil
from depiction.parallel_ops import ParallelConfig, ReadSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlReader


@dataclass
class IsotopePatternMatcher:
    """Compares a concrete isotope pattern with a reference one, e.g. an averagine pattern"
    The results of the computation are cached internally and the size of the cache can be specified with cache_size.
    """

    # TODO experimental code

    cache_size: int
    cache_tolerance: float = 0.5

    def __post_init__(self):
        self._pattern_cache = OrderedDict()  # type: OrderedDict[float, tuple[NDArray[float], NDArray[float]]]

    def get_averagine_pattern(self, mass: float) -> tuple[NDArray[float], NDArray[float]]:
        """Returns the averagine isotope pattern for the specified mass value and
        returns the mz and intensity arrays. If a result within `cache_tolerance` is already cached,
        it will be returned instead of computing the pattern (note that a correction will be applied such that the first
        element of the cached pattern will have the specified mass)."""
        closest_cache_mass = self._find_closest_cache_mass(mass)
        if closest_cache_mass:
            self._pattern_cache.move_to_end(closest_cache_mass)
            mz_arr, int_arr = self._pattern_cache[closest_cache_mass]
            # since there is potentially some small discrepancy, adjust the mz values
            # (note: this might not be fully precise)
            return mz_arr - (mz_arr[0] - mass), int_arr
        else:
            mz_arr, int_arr = self._compute_averagine_pattern(mass=mass)
            self._pattern_cache[mass] = (mz_arr, int_arr)
            self._ensure_cache_size()
            return mz_arr, int_arr

    def compute_averagine_agreement(
        self,
        mz_peaks: NDArray[float],
        int_peaks: NDArray[float],
        n_limit: int,
        distance_tolerance: float,
    ) -> tuple[float, int]:
        """Returns the cosine similarity between the averagine pattern and the specified peaks.
        It also returns the number of peaks that were actually used for the comparison.
        :param mz_peaks: the mz values of the peaks
        :param int_peaks: the intensity values of the peaks
        :param n_limit: the maximum number of peaks to use for the comparison
        :param distance_tolerance: the tolerance for the mz values of the peaks, if peaks are not aligned between the
            averagine pattern and the peaks, the cosine similarity will be 0.0.
        :return:
            - cosine_similarity: the cosine similarity between the averagine pattern and the peaks
            - n_align: the number of peaks that were actually used for the comparison
        """
        # Compute the averagine pattern starting at the first peak mass.
        mz_avg, int_avg = self.get_averagine_pattern(mass=mz_peaks[0])

        # Compute the spectra agreement.
        return self.compute_spectra_agreement(
            spectrum_1=(mz_avg, int_avg),
            spectrum_2=(mz_peaks, int_peaks),
            n_limit=n_limit,
            distance_tolerance=distance_tolerance,
        )

    def compute_averagine_agreement_at_positions(
        self,
        mz_peaks: NDArray[float],
        int_peaks: NDArray[float],
        idx_positions: NDArray[int],
        n_limit: int,
        distance_tolerance: float,
    ) -> tuple[NDArray[float], NDArray[int]]:
        agreement_scores = np.zeros(len(idx_positions))
        agreement_lengths = np.zeros(len(idx_positions), dtype=int)

        for i_peak, idx_peak in enumerate(idx_positions):
            (
                agreement_scores[i_peak],
                agreement_lengths[i_peak],
            ) = self.compute_averagine_agreement(
                mz_peaks=mz_peaks[idx_peak:],
                int_peaks=int_peaks[idx_peak:],
                n_limit=n_limit,
                distance_tolerance=distance_tolerance,
            )

        return agreement_scores, agreement_lengths

    def compute_averagine_agreement_at_mz_positions_for_file(
        self,
        read_file: ImzmlReadFile,
        parallel_config: ParallelConfig,
        peak_picker: BasicPeakPicker,
        n_limit: int,
        distance_tolerance: float,
        mz_positions_of_interest: NDArray[float],
        spectra_ids: Optional[list[int]] = None,
    ) -> list[tuple[NDArray[float], NDArray[int]]]:
        # TODO possibly move this method in the future (since it mixes peak_picker into this class)

        def operation_file(reader: ImzmlReader, spectra_ids: list[int]) -> list[tuple[NDArray[float], NDArray[int]]]:
            results = []
            for spectrum_id in spectra_ids:
                mz_arr, int_arr = reader.get_spectrum(spectrum_id)
                mz_peaks, int_peaks = peak_picker.pick_peaks(mz_arr, int_arr)

                # determine the best indices for the mz positions of interest
                idx_positions = NumpyUtil.search_sorted_closest(full_array=mz_peaks, values=mz_positions_of_interest)

                results.append(
                    self.compute_averagine_agreement_at_positions(
                        mz_peaks=mz_peaks,
                        int_peaks=int_peaks,
                        idx_positions=idx_positions,
                        n_limit=n_limit,
                        distance_tolerance=distance_tolerance,
                    )
                )
            return results

        parallelize = ReadSpectraParallel.from_config(parallel_config)
        return parallelize.map_chunked(
            read_file=read_file,
            operation=operation_file,
            reduce_fn=parallelize.reduce_concat,
            spectra_indices=spectra_ids,
        )

    @classmethod
    def compute_spectra_agreement(
        cls,
        spectrum_1: tuple[NDArray[float], NDArray[float]],
        spectrum_2: tuple[NDArray[float], NDArray[float]],
        n_limit: int,
        distance_tolerance: float,
    ) -> tuple[float, int]:
        mz_arr_1, int_arr_1 = spectrum_1
        mz_arr_2, int_arr_2 = spectrum_2

        # Check alignment of mz values
        # TODO in the future it would be better to actually match/align in case one of the spectra is more sparse
        #      but for now this method will basically fail as soon as there is an extra peak,
        #      maybe there would still be a use case for both methods then
        n_compare = min(len(mz_arr_1), len(mz_arr_2))
        is_align = np.isclose(mz_arr_1[:n_compare], mz_arr_2[:n_compare], atol=distance_tolerance)
        idx_first_not_align = len(is_align) if np.all(is_align) else np.where(~is_align)[0][0]

        # Select the peaks to be compared
        n_compare = min(idx_first_not_align, n_limit, n_compare)

        # Perform the comparison
        if n_compare < 2:
            return 0.0, n_compare
        else:
            return (
                cls.cosine_similarity(int_arr_1[:n_compare], int_arr_2[:n_compare]),
                n_compare,
            )

    @staticmethod
    def cosine_similarity(x_vec: NDArray[float], y_vec: NDArray[float]) -> float:
        """Returns the cosine similarity between two vectors."""
        # TODO currently this will print a message when dividing by approximately zero
        return np.dot(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))

    def _find_closest_cache_mass(self, mass: float) -> Optional[float]:
        """Tries to find the closest mass in the cache to the specified mass value. If no value exists, or it is not
        within the specified tolerance, None is returned."""
        if not self._pattern_cache:
            return None
        sorted_cache_masses = np.sort(list(self._pattern_cache.keys()))
        closest_index = np.argmin(np.abs(sorted_cache_masses - mass))
        closest_cache_mass = sorted_cache_masses[closest_index]
        if np.abs(closest_cache_mass - mass) <= self.cache_tolerance:
            return closest_cache_mass
        else:
            return None

    def _ensure_cache_size(self) -> None:
        """Removes the least recently used cache entries if the cache size is exceeded such that the cache size is
        within the specified limit again."""
        if not self.cache_size > 0:
            return
        while len(self._pattern_cache) > self.cache_size:
            self._pattern_cache.popitem(last=False)

    @staticmethod
    def _compute_averagine_pattern(
        mass: float,
    ) -> tuple[NDArray[float], NDArray[float]]:
        """Computes the averagine isotope pattern for the specified mass value and returns the mz and intensity arrays.
        If possible, use get_averagine_pattern method which will invoke a cached version of this function.
        """
        return alphapept.chem.mass_to_dist(mass, alphapept.constants.averagine_aa, alphapept.constants.isotopes)
