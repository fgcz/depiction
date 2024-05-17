import numpy as np
from numpy.typing import NDArray

from ionmapper.peak_picking.basic_peak_picker import BasicPeakPicker
from ionmapper.parallel_ops import ParallelConfig, ReadSpectraParallel
from ionmapper.persistence import ImzmlReadFile, ImzmlReader
from ionmapper.image.sparse_image_2d import SparseImage2d


class GenerateIsotopicImages:
    def __init__(self, max_ref_peak_distance: float, parallel_config: ParallelConfig) -> None:
        self._peak_picker = BasicPeakPicker(smooth_sigma=1.0, min_prominence=0.1)
        self._max_ref_peak_distance = max_ref_peak_distance
        self._parallel_config = parallel_config

    def generate_isotopic_images_for_file(
        self,
        input_file: ImzmlReadFile,
        mz_ref_list: list[float],
        n_isotopes: int,
    ):
        def task(reader: ImzmlReader, spectra_indices: list[int]):
            return self._generate_isotopic_image_values(
                reader=reader,
                mz_ref_list=mz_ref_list,
                n_isotopes=n_isotopes,
                spectra_indices=spectra_indices,
                max_ref_peak_distance=self._max_ref_peak_distance,
                peak_picker=self._peak_picker.clone(),
            )

        parallelize = ReadSpectraParallel.from_config(self._parallel_config)
        values = parallelize.map_chunked(
            read_file=input_file,
            operation=task,
            reduce_fn=lambda chunks: np.concatenate(chunks, axis=0),
        )
        return SparseImage2d(values=values, coordinates=input_file.coordinates_2d)

    def generate_isotopic_image(
        self,
        input_file: ImzmlReadFile,
        mz_ref_list: list[float],
        n_isotopes: int,
        spectra_indices: list[int],
    ) -> SparseImage2d:
        with input_file.reader() as reader:
            values = self._generate_isotopic_image_values(
                reader=reader,
                mz_ref_list=mz_ref_list,
                n_isotopes=n_isotopes,
                spectra_indices=spectra_indices,
                max_ref_peak_distance=self._max_ref_peak_distance,
                peak_picker=self._peak_picker,
            )
        coordinates = input_file.coordinates_2d[np.asarray(spectra_indices)]
        return SparseImage2d(values=values, coordinates=coordinates)

    @classmethod
    def _generate_isotopic_image_values(
        cls,
        reader: ImzmlReader,
        mz_ref_list: list[float],
        n_isotopes: int,
        spectra_indices: list[int],
        max_ref_peak_distance: float,
        peak_picker: BasicPeakPicker,
    ):
        n_spectra = len(spectra_indices)
        n_mz_ref = len(mz_ref_list)
        result = np.zeros((n_spectra, n_mz_ref), float)

        for idx_spectrum, spectrum_id in enumerate(spectra_indices):
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)

            # pick peaks
            peak_idx = peak_picker.pick_peaks_index(mz_arr=mz_arr, int_arr=int_arr)
            if len(peak_idx) == 0:
                # default to zero intensity
                continue
            peak_mz = mz_arr[peak_idx]
            peak_int = int_arr[peak_idx]

            result[idx_spectrum, :] = cls._compute_isotopic_intensity_for_spectrum(
                mz_ref_list=mz_ref_list,
                n_isotopes=n_isotopes,
                peak_int=peak_int,
                peak_mz=peak_mz,
                max_ref_peak_distance=max_ref_peak_distance,
            )

        return result

    @staticmethod
    def _compute_isotopic_intensity_for_spectrum(
        mz_ref_list: list[float],
        n_isotopes: int,
        peak_int: NDArray[float],
        peak_mz: NDArray[float],
        max_ref_peak_distance: float,
    ) -> NDArray[float]:
        n_peaks = len(peak_int)
        n_mz_ref = len(mz_ref_list)

        result = np.zeros(n_mz_ref)

        for i_mz_ref, mz_ref in enumerate(mz_ref_list):
            # now try to find the position of the closest peak to mz_ref
            ref_peak_idx = np.argmin(np.abs(peak_mz - mz_ref))
            ref_peak_dist = np.abs(peak_mz[ref_peak_idx] - mz_ref)

            # if it's too close, skip this one
            if ref_peak_dist > max_ref_peak_distance:
                # default to zero intensity
                continue

            # determine the isotopic peaks
            isotopic_peak_indices = np.arange(ref_peak_idx, min(ref_peak_idx + n_isotopes, n_peaks))

            # remove peaks that are too far distanced from the previous peak
            # TODO make the threshold customizable?
            inter_isotopic_peak_distances = np.diff(peak_mz[isotopic_peak_indices])
            distance_too_big = inter_isotopic_peak_distances > 1.3
            if np.any(distance_too_big):
                # keep only the acceptable ones
                isotopic_peak_indices = isotopic_peak_indices[: np.argmax(distance_too_big) + 1]
                if len(isotopic_peak_indices) == 0:
                    # default to zero intensity
                    continue

            # and now sum the intensities
            result[i_mz_ref] = np.sum(peak_int[isotopic_peak_indices])

        return result
