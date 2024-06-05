import shutil
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from loguru import logger
from typer import Option

from depiction.calibration.perform_calibration import PerformCalibration
from depiction.calibration.spectrum.calibration_method_global_constant_shift import CalibrationMethodGlobalConstantShift
from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlModeEnum
from depiction_targeted_preproc.pipeline_config import model
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def calibrate_make_fair_comparison(
    input_imzml_path: Annotated[Path, Option()],
    config_path: Annotated[Path, Option()],
    mass_list_path: Annotated[Path, Option()],
    output_imzml_path: Annotated[Path, Option()],
) -> None:
    # Read the config
    config = PipelineParameters.parse_yaml(config_path)
    mass_list = pl.read_csv(mass_list_path)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)

    # Determine if we should simply copy the file or if we should perform some GlobalConstantShift calibration
    if isinstance(config.calibration, model.CalibrationRegressShift) or config.calibration is None:
        # no additional calibration requested, simply copy the file
        logger.info("Copying input file to output")
        shutil.copy(input_imzml_path, output_imzml_path)
        shutil.copy(input_imzml_path.with_suffix(".ibd"), output_imzml_path.with_suffix(".ibd"))
    else:
        logger.info("Performing GlobalConstantShift calibration")
        calibration = CalibrationMethodGlobalConstantShift(ref_mz_arr=mass_list["mass"].to_numpy())
        perform_calibration = PerformCalibration(
            calibration=calibration,
            parallel_config=parallel_config,
            output_store=None,
            coefficient_output_file=None,
        )
        perform_calibration.calibrate_image(
            read_peaks=ImzmlReadFile(input_imzml_path),
            write_file=ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.PROCESSED),
        )


if __name__ == "__main__":
    typer.run(calibrate_make_fair_comparison)
