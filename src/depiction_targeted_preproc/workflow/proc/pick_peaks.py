import shutil
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlModeEnum, ImzmlWriteFile, ImzmlReadFile
from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned
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

    # TODO configurable filtering
    peak_filtering = FilterNHighestIntensityPartitioned(max_count=200, n_partitions=8)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)

    match config.peak_picker:
        case None:
            logger.info("Peak picking is deactivated")
            shutil.copy(input_imzml_path, output_imzml_path)
            shutil.copy(input_imzml_path.with_suffix(".ibd"), output_imzml_path.with_suffix(".ibd"))
            return
        case model.PeakPickerBasicInterpolated() as peak_picker_config:
            peak_picker = BasicInterpolatedPeakPicker(
                min_prominence=peak_picker_config.min_prominence,
                min_distance=peak_picker_config.min_distance,
                min_distance_unit=peak_picker_config.min_distance_unit,
                peak_filtering=peak_filtering,
            )
        case model.PeakPickerMSPeakPicker() as peak_picker_config:
            peak_picker = MSPeakPicker(fit_type=peak_picker_config.fit_type, peak_filtering=peak_filtering)
        case _:
            raise ValueError(f"Unsupported peak picker type: {config.peak_picker.peak_picker_type}")

    # TODO correctly detect files which are already picked

    pick_peaks = PickPeaks(peak_picker=peak_picker, parallel_config=parallel_config)
    read_file = ImzmlReadFile(input_imzml_path)
    write_file = ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.PROCESSED)
    pick_peaks.evaluate_file(read_file, write_file)


def main() -> None:
    typer.run(proc_pick_peaks)


if __name__ == "__main__":
    main()
