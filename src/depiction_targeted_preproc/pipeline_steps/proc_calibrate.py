import logging
import shutil
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from depiction.calibration.perform_calibration import PerformCalibration
from depiction.calibration.spectrum.calibration_pipeline_chemical_peptide_noise import (
    CalibrationPipelineChemicalPeptideNoise,
)
from depiction.calibration.spectrum.calibration_pipeline_regress_shift import CalibrationPipelineRegressShift
from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlModeEnum
from depiction_targeted_preproc.pipeline_config import model
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def get_calibration_from_config(mass_list: pl.DataFrame, config: PipelineParameters):
    parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)
    match config.calibration:
        case model.CalibrationRegressShift() as calib_config:
            return CalibrationPipelineRegressShift(
                ref_mz_arr=mass_list["mass"].to_numpy(),
                max_distance=calib_config.max_distance,
                max_distance_unit=calib_config.max_distance_unit,
                model_type=calib_config.reg_model_type,
                model_unit=calib_config.reg_model_unit,
                input_smoothing_activated=calib_config.input_smoothing_activated,
                input_smoothing_kernel_size=calib_config.input_smoothing_kernel_size,
                input_smoothing_kernel_std=calib_config.input_smoothing_kernel_std,
                min_points=calib_config.min_points,
            )
        case model.CalibrationChemicalPeptideNoise() as calib_config:
            return CalibrationPipelineChemicalPeptideNoise(
                n_mass_intervals=calib_config.n_mass_intervals,
                interpolation_mode=calib_config.interpolation_mode,
                parallel_config=parallel_config,
                use_ppm_space=calib_config.use_ppm_space,
            )
        case _:
            raise NotImplementedError("should be unreachable")


def proc_calibrate(
    input_imzml_path: Annotated[Path, typer.Option()],
    config_path: Annotated[Path, typer.Option()],
    mass_list_path: Annotated[Path, typer.Option()],
    output_imzml_path: Annotated[Path, typer.Option()],
    output_calib_data_path: Annotated[Path, typer.Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    mass_list = pl.read_csv(mass_list_path)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)
    match config.calibration:
        case None:
            print("No calibration requested")
            shutil.copy(input_imzml_path, output_imzml_path)
            shutil.copy(input_imzml_path.with_suffix(".ibd"), output_imzml_path.with_suffix(".ibd"))
        case model.CalibrationRegressShift() | model.CalibrationChemicalPeptideNoise():
            calibration = get_calibration_from_config(mass_list, config)
            perform_calibration = PerformCalibration(
                calibration=calibration,
                parallel_config=parallel_config,
                output_store=None,
                coefficient_output_file=output_calib_data_path,
            )
            perform_calibration.calibrate_image(
                read_peaks=ImzmlReadFile(input_imzml_path),
                write_file=ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.PROCESSED),
                # TODO
                read_full=None,
            )
        case _:
            raise NotImplementedError("should be unreachable")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(proc_calibrate)


if __name__ == "__main__":
    main()
