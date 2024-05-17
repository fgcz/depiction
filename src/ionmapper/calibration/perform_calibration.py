# TODO very experimental first version
# TODO needs better name (but like this for easier distinguishing)
import logging
from typing import Protocol, Optional
import h5py
import numpy as np
from numpy.typing import NDArray

from ionmapper.parallel_ops import ParallelConfig, ReadSpectraParallel, WriteSpectraParallel
from ionmapper.peak_picking import BasicInterpolatedPeakPicker
from ionmapper.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlReader, ImzmlWriter


class CalibrationType(Protocol):
    def preprocess_spectrum(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> NDArray[float]:
        pass

    def process_coefficients(self, all_coefficients: NDArray[float], coordinates_2d: NDArray[int]) -> NDArray[float]:
        pass

    def calibrate_spectrum(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], coefficients: NDArray[float]
    ) -> NDArray[float]:
        pass


class PerformCalibration:
    def __init__(
        self,
        calibration: CalibrationType,
        parallel_config: ParallelConfig,
        output_store: h5py.Group | None = None,
        # TODO this should be deprecated (even though it might currently have some perf benefits)
        peak_picker: Optional[BasicInterpolatedPeakPicker] = None,
    ) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config
        self._peak_picker = peak_picker
        self._output_store = output_store

    def calibrate_image(
        self, read_peaks: ImzmlReadFile, write_file: ImzmlWriteFile, read_full: Optional[ImzmlReadFile] = None
    ) -> None:
        read_parallel = ReadSpectraParallel.from_config(self._parallel_config)
        logger = logging.getLogger(__name__)
        logger.info("Computing initial coefficients...")
        coefficients = read_parallel.map_chunked(
            read_file=read_peaks,
            operation=self._get_initial_coefficients,
            bind_args=dict(
                calibration=self._calibration,
                peak_picker=self._peak_picker,
            ),
            reduce_fn=lambda chunks: np.concatenate(chunks, axis=0),
        )
        self._register_coefficients(read_peaks.coordinates_2d, label="coordinates_2d")
        self._register_coefficients(coefficients, label="coef_initial")
        logger.info("Processing coefficients...")
        coefficients = self._calibration.process_coefficients(
            all_coefficients=coefficients, coordinates_2d=read_peaks.coordinates_2d
        )
        self._register_coefficients(coefficients, label="coef_processed")
        logger.info("Calibrating spectra...")
        write_parallel = WriteSpectraParallel.from_config(self._parallel_config)
        write_parallel.map_chunked_to_file(
            read_file=read_full or read_peaks,
            write_file=write_file,
            operation=self._calibrate_spectra,
            bind_args=dict(
                calibration=self._calibration,
                coefficients=coefficients,
            ),
        )

    @staticmethod
    def _get_initial_coefficients(
        reader: ImzmlReader,
        spectra_indices: list[int],
        calibration: CalibrationType,
        peak_picker: Optional[BasicInterpolatedPeakPicker],
    ):
        collect = []
        for spectrum_id in spectra_indices:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            if peak_picker:
                mz_arr, int_arr = peak_picker.pick_peaks(mz_arr, int_arr)
            preprocessed = calibration.preprocess_spectrum(peak_mz_arr=mz_arr, peak_int_arr=int_arr)
            collect.append(preprocessed)
        return np.stack(collect, axis=0)

    @staticmethod
    def _calibrate_spectra(
        reader: ImzmlReader,
        spectra_indices: list[int],
        writer: ImzmlWriter,
        calibration: CalibrationType,
        coefficients: NDArray[float],
    ) -> None:
        for spectrum_id in spectra_indices:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            calibrated_mz_arr = calibration.calibrate_spectrum(
                spectrum_mz_arr=mz_arr, spectrum_int_arr=int_arr, coefficients=coefficients[spectrum_id]
            )
            writer.add_spectrum(calibrated_mz_arr, int_arr, coords)

    def _register_coefficients(self, coefficients: NDArray[float], label: str) -> None:
        """Registers the coefficients, writing them to the output hdf5 group if configured."""
        if self._output_store:
            self._output_store.create_dataset(label, data=coefficients)
