from __future__ import annotations

from pathlib import Path
from typing import Literal, Annotated

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from depiction.calibration.perform_calibration import PerformCalibration
from depiction.calibration.spectrum.calibration_method_chemical_peptide_noise import (
    CalibrationMethodChemicalPeptideNoise,
)
from depiction.calibration.spectrum.calibration_method_global_constant_shift import CalibrationMethodGlobalConstantShift
from depiction.calibration.spectrum.calibration_method_mcc import CalibrationMethodMassClusterCenterModel
from depiction.calibration.spectrum.calibration_method_regress_shift import CalibrationMethodRegressShift
from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile


class CalibrationRegressShiftConfig(BaseModel):
    calibration_method: Literal["RegressShift"] = "RegressShift"

    max_distance: float
    # TODO make explicit
    max_distance_unit: str
    # TODO make explicit
    reg_model_type: str
    # TODO make explicit
    reg_model_unit: str
    input_smoothing_activated: bool
    input_smoothing_kernel_size: int = 27
    input_smoothing_kernel_std: float = 10.0
    min_points: int = 3


class CalibrationChemicalPeptideNoiseConfig(BaseModel):
    calibration_method: Literal["ChemicalPeptideNoise"] = "ChemicalPeptideNoise"

    n_mass_intervals: int
    interpolation_mode: Literal["linear", "cubic_spline", "refit_linear"] = "linear"
    use_ppm_space: bool = False


class CalibrationMCCConfig(BaseModel):
    calibration_method: Literal["MCC"] = "MCC"

    coef_smoothing_activated: bool
    coef_smoothing_kernel_size: int = 27
    coef_smoothing_kernel_std: float = 10.0


class CalibrationConstantGlobalShiftConfig(BaseModel):
    calibration_method: Literal["ConstantGlobalShift"] = "ConstantGlobalShift"


class CalibrationConfig(BaseModel, use_enum_values=True, validate_default=True):
    method: (
        Annotated[
            CalibrationRegressShiftConfig
            | CalibrationChemicalPeptideNoiseConfig
            | CalibrationMCCConfig
            | CalibrationConstantGlobalShiftConfig,
            Field(discriminator="calibration_method"),
        ]
        | None
    )

    n_jobs: int | None = None


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
        case _:
            raise NotImplementedError("should be unreachable")


def calibrate(
    config: CalibrationConfig,
    input_file: ImzmlReadFile,
    output_file: ImzmlWriteFile,
    mass_list: Path | None = None,
    coefficient_output_path: Path | None = None,
) -> None:
    if config.method is None:
        logger.info("No calibration requested")
        input_file.copy_to(output_file.imzml_file)
    else:
        calibration = get_calibration_instance(config=config, mass_list=mass_list)
        parallel_config = ParallelConfig(n_jobs=config.n_jobs)
        logger.info("Using calibration method: {calibration}", calibration=calibration)

        # TODO is output_store still used
        perform_calibration = PerformCalibration(
            calibration=calibration,
            parallel_config=parallel_config,
            output_store=None,
            coefficient_output_file=coefficient_output_path,
        )
        perform_calibration.calibrate_image(
            read_peaks=input_file,
            write_file=output_file,
            # TODO:make it possible to customize this
            read_full=None,
        )
