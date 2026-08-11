from depiction.calibration.methods.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction_io.types import GenericReadFile, GenericWriteFile, GenericReader, GenericWriter


class ApplyModels:
    """Apply calibration models to mass spectra in parallel.

    This class provides functionality to apply calibration models to a set of mass spectra,
    handling the parallel processing and file I/O operations.

    :param calibration: The calibration method to apply to the spectra
    :param parallel_config: Configuration for parallel processing operations
    """

    def __init__(self, calibration: CalibrationMethod, parallel_config: ParallelConfig) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config

    def write_to_file(
        self, read_file: GenericReadFile, write_file: GenericWriteFile, all_model_coefs: MultiChannelImage
    ) -> None:
        """Writes calibrated spectra to an output file.

        Reads spectra from the input file, applies calibration models, and writes the
        calibrated results to the output file in parallel.

        :param read_file: Input file containing uncalibrated spectra
        :param write_file: Output file to write calibrated spectra
        :param all_model_coefs: Model coefficients for all spectra
        """
        write_parallel = WriteSpectraParallel.from_config(self._parallel_config)
        write_parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self.calibrate_spectra,
            bind_args=dict(
                calibration=self._calibration,
                all_model_coefs=all_model_coefs,
            ),
        )

    @staticmethod
    def calibrate_spectra(
        reader: GenericReader,
        spectra_indices: list[int],
        writer: GenericWriter,
        calibration: CalibrationMethod,
        all_model_coefs: MultiChannelImage,
    ) -> None:
        """Calibrates a batch of spectra and writes to output writer.

        :param reader: Reader for accessing uncalibrated spectra
        :param spectra_indices: List of spectrum indices to process
        :param writer: Writer for saving calibrated spectra
        :param calibration: Calibration method to apply
        :param all_model_coefs: Model coefficients for all spectra
        """
        for spectrum_id in spectra_indices:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            # Index by coordinate: the flat model order is (y, x) row-major, which is not the
            # spectrum order unless the file happens to be acquired that way.
            features = all_model_coefs.data_spatial.sel(x=coords[0], y=coords[1])
            calib_mz_arr, calib_int_arr = calibration.apply_spectrum_model(
                spectrum_mz_arr=mz_arr, spectrum_int_arr=int_arr, model_coef=features
            )
            writer.add_spectrum(calib_mz_arr, calib_int_arr, coords)
