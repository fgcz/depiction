from loguru import logger
from pathlib import Path
from typing import Optional

from depiction.calibration.apply.apply_models import ApplyModels
from depiction.calibration.apply.extract_features import ExtractFeatures
from depiction.calibration.apply.fit_models import FitModels
from depiction.calibration.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig
from depiction.persistence.types import GenericReadFile, GenericWriteFile


class CalibrateImage:
    def __init__(
        self,
        calibration: CalibrationMethod,
        parallel_config: ParallelConfig,
        coefficient_output_file: Path | None = None,
    ) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config
        self._coefficient_output_file = coefficient_output_file

    def calibrate_image(
        self, read_peaks: GenericReadFile, write_file: GenericWriteFile, read_full: Optional[GenericReadFile] = None
    ) -> None:
        read_full = read_full or read_peaks

        logger.info("Extracting all features...")
        all_features = ExtractFeatures(self._calibration, self._parallel_config).get_image(read_peaks)
        self._write_data_array(all_features, group="features_raw")

        logger.info("Preprocessing features...")
        all_features = self._calibration.preprocess_image_features(all_features=all_features)
        self._write_data_array(all_features, group="features_processed")

        logger.info("Fitting models...")
        model_coefs = FitModels(self._calibration, self._parallel_config).get_image(all_features)
        self._write_data_array(model_coefs, group="model_coefs")

        logger.info("Applying models...")
        ApplyModels(self._calibration, self._parallel_config).write_to_file(
            read_file=read_full, write_file=write_file, all_model_coefs=model_coefs
        )

    def _write_data_array(self, image: MultiChannelImage, group: str) -> None:
        """Exports the given image into a HDF5 group of the coefficient output file (if specified)."""
        if not self._coefficient_output_file:
            return
        image.write_hdf5(path=self._coefficient_output_file, mode="a", group=group)
