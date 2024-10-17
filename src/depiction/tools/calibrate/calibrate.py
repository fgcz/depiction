from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from depiction.calibration.perform_calibration import PerformCalibration
from depiction.calibration.spectrum.calibration_method_chemical_peptide_noise import (
    CalibrationMethodChemicalPeptideNoise,
)
from depiction.calibration.spectrum.calibration_method_dummy import CalibrationMethodDummy
from depiction.calibration.spectrum.calibration_method_global_constant_shift import CalibrationMethodGlobalConstantShift
from depiction.calibration.spectrum.calibration_method_mcc import CalibrationMethodMassClusterCenterModel
from depiction.calibration.spectrum.calibration_method_regress_shift import CalibrationMethodRegressShift
from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile
from depiction.tools.calibrate.config import (
    CalibrationRegressShiftConfig,
    CalibrationChemicalPeptideNoiseConfig,
    CalibrationMCCConfig,
    CalibrationConstantGlobalShiftConfig,
    CalibrationConfig,
)
from depiction.tools.filter_peaks.filter_peaks import get_peak_filter


def extract_reference_masses(mass_list: Path) -> NDArray[float]:
    if not mass_list.exists():
        raise RuntimeError(
            f"Mass list file {mass_list} does not exist but is required for the calibration method. "
            "Please provide a mass list or pick an untargeted calibration method."
        )
    df = pl.read_csv(mass_list)
    # TODO maybe revisit this in the future, or add a warning if the panel was not sorted before!
    return np.sort(df["mass"].to_numpy())


# TODO better name, return type
def get_calibration_instance(config: CalibrationConfig, mass_list: Path | None):
    match config.method:
        case CalibrationRegressShiftConfig():
            peak_filtering = get_peak_filter(config.method.peak_filtering) if config.method.peak_filtering else None
            return CalibrationMethodRegressShift(
                ref_mz_arr=extract_reference_masses(mass_list),
                max_distance=config.method.max_distance,
                max_distance_unit=config.method.max_distance_unit,
                model_type=config.method.reg_model_type,
                model_unit=config.method.reg_model_unit,
                input_smoothing_activated=config.method.input_smoothing_activated,
                input_smoothing_kernel_size=config.method.input_smoothing_kernel_size,
                input_smoothing_kernel_std=config.method.input_smoothing_kernel_std,
                min_points=config.method.min_points,
                peak_filtering=peak_filtering,
            )
        case CalibrationConstantGlobalShiftConfig():
            return CalibrationMethodGlobalConstantShift(
                ref_mz_arr=extract_reference_masses(mass_list),
            )
        case CalibrationChemicalPeptideNoiseConfig():
            return CalibrationMethodChemicalPeptideNoise(
                n_mass_intervals=config.method.n_mass_intervals,
                interpolation_mode=config.method.interpolation_mode,
                use_ppm_space=config.method.use_ppm_space,
            )
        case CalibrationMCCConfig():
            return CalibrationMethodMassClusterCenterModel(
                model_smoothing_activated=config.method.coef_smoothing_activated,
                model_smoothing_kernel_size=config.method.coef_smoothing_kernel_size,
                model_smoothing_kernel_std=config.method.coef_smoothing_kernel_std,
            )
        case None:
            return CalibrationMethodDummy()
        case _:
            raise NotImplementedError("should be unreachable")


def calibrate(
    config: CalibrationConfig,
    input_file: ImzmlReadFile,
    output_file: ImzmlWriteFile,
    mass_list: Path | None = None,
    coefficient_output_path: Path | None = None,
) -> None:
    calibration = get_calibration_instance(config=config, mass_list=mass_list)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs)
    logger.info("Using calibration method: {calibration}", calibration=calibration)

    perform_calibration = PerformCalibration(
        calibration=calibration,
        parallel_config=parallel_config,
        coefficient_output_file=coefficient_output_path,
    )
    perform_calibration.calibrate_image(
        read_peaks=input_file,
        write_file=output_file,
        # TODO:make it possible to customize this
        read_full=None,
    )
