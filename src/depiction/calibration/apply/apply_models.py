from depiction.calibration.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence.types import GenericReadFile, GenericWriteFile, GenericReader, GenericWriter


class ApplyModels:
    def __init__(self, calibration: CalibrationMethod, parallel_config: ParallelConfig) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config

    def write_to_file(
        self, read_file: GenericReadFile, write_file: GenericWriteFile, all_model_coefs: MultiChannelImage
    ) -> None:
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
        for spectrum_id in spectra_indices:
            # TODO sanity check the usage of i as spectrum_id (i.e. check the coords!)
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            features = all_model_coefs.data_flat.isel(i=spectrum_id)
            calib_mz_arr, calib_int_arr = calibration.apply_spectrum_model(
                spectrum_mz_arr=mz_arr, spectrum_int_arr=int_arr, model_coef=features
            )
            writer.add_spectrum(calib_mz_arr, calib_int_arr, coords)
