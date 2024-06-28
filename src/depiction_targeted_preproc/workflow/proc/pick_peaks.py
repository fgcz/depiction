import shutil
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlModeEnum, ImzmlWriteFile, ImzmlReadFile
from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned, PeakFilteringType
from depiction.spectrum.peak_picking import BasicInterpolatedPeakPicker
from depiction.spectrum.peak_picking.ms_peak_picker_wrapper import MSPeakPicker
from depiction.tools.pick_peaks import PickPeaks
from depiction_targeted_preproc.pipeline_config import model
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def proc_pick_peaks(
    input_imzml_path: Annotated[Path, typer.Option()],
    config_path: Annotated[Path, typer.Option()],
    output_imzml_path: Annotated[Path, typer.Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    read_file = ImzmlReadFile(input_imzml_path)

    if config.peak_picker is None or (not config.force_peak_picker and read_file.imzml_mode == ImzmlModeEnum.PROCESSED):
        copy_without_picking(input_imzml_path=input_imzml_path, output_imzml_path=output_imzml_path)
    else:
        perform_peak_picking(config=config, read_file=read_file, output_imzml_path=output_imzml_path)


def get_peak_filtering(config: model.PeakFiltering) -> PeakFilteringType | None:
    match config:
        case model.FilterNHighestIntensityPartitioned() as c:
            return FilterNHighestIntensityPartitioned(max_count=c.max_count, n_partitions=c.n_partitions)
        case None:
            return None
        case _:
            raise ValueError(f"Unsupported peak filtering type: {config}")


def perform_peak_picking(config: PipelineParameters, read_file: ImzmlReadFile, output_imzml_path: Path) -> None:
    peak_filtering = get_peak_filtering(config.peak_filtering)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)
    match config.peak_picker:
        case model.PeakPickerBasicInterpolated() as peak_picker_config:
            peak_picker = BasicInterpolatedPeakPicker(
                min_prominence=peak_picker_config.min_prominence,
                min_distance=peak_picker_config.min_distance,
                min_distance_unit=peak_picker_config.min_distance_unit,
                peak_filtering=peak_filtering,
            )
        case model.PeakPickerMSPeakPicker() as peak_picker_config:
            peak_picker = MSPeakPicker(fit_type=peak_picker_config.fit_type, peak_filtering=peak_filtering)
        case model.PeakPickerFindMFPy() as peak_picker_config:
            # NOTE: importing this here since it has non-standard dependencies
            from depiction.spectrum.peak_picking.findmf_peak_picker import FindMFPeakpicker

            peak_picker = FindMFPeakpicker(
                resolution=peak_picker_config.resolution,
                width=peak_picker_config.width,
                int_width=peak_picker_config.int_width,
                int_threshold=peak_picker_config.int_threshold,
                area=peak_picker_config.area,
                max_peaks=peak_picker_config.max_peaks,
            )
        case _:
            raise ValueError(f"Unsupported peak picker type: {config.peak_picker.peak_picker_type}")
    # TODO correctly detect files which are already picked
    pick_peaks = PickPeaks(peak_picker=peak_picker, parallel_config=parallel_config)
    write_file = ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.PROCESSED)
    pick_peaks.evaluate_file(read_file, write_file)


def copy_without_picking(input_imzml_path: Path, output_imzml_path: Path) -> None:
    logger.info("Peak picking is deactivated")
    shutil.copy(input_imzml_path, output_imzml_path)
    shutil.copy(input_imzml_path.with_suffix(".ibd"), output_imzml_path.with_suffix(".ibd"))


def main() -> None:
    typer.run(proc_pick_peaks)


if __name__ == "__main__":
    main()
