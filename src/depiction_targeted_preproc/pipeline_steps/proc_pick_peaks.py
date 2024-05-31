import shutil
from pathlib import Path
from typing import Annotated

import typer
from depiction.parallel_ops import ParallelConfig
from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned
from depiction.spectrum.peak_picking import BasicInterpolatedPeakPicker
from depiction.persistence import ImzmlModeEnum, ImzmlWriteFile, ImzmlReadFile
from depiction.tools.pick_peaks import PickPeaks

from depiction_targeted_preproc.pipeline_config import model
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def proc_pick_peaks(
    input_imzml_path: Annotated[Path, typer.Option()],
    config_path: Annotated[Path, typer.Option()],
    output_imzml_path: Annotated[Path, typer.Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    match config.peak_picker:
        case None:
            print("Peak picking is deactivated")
            shutil.copy(input_imzml_path, output_imzml_path)
            shutil.copy(input_imzml_path.with_suffix(".ibd"), output_imzml_path.with_suffix(".ibd"))
        case model.PeakPickerBasicInterpolated() as peak_picker_config:
            peak_picker = BasicInterpolatedPeakPicker(
                min_prominence=peak_picker_config.min_prominence,
                min_distance=peak_picker_config.min_distance,
                min_distance_unit=peak_picker_config.min_distance_unit,
                # TODO configurable
                peak_filtering=FilterNHighestIntensityPartitioned(max_count=200, n_partitions=8),
            )
            parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)
            pick_peaks = PickPeaks(
                peak_picker=peak_picker,
                parallel_config=parallel_config,
            )
            read_file = ImzmlReadFile(input_imzml_path)
            write_file = ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.PROCESSED)
            pick_peaks.evaluate_file(read_file, write_file)
        case _:
            raise ValueError(f"Unsupported peak picker type: {config.peak_picker.peak_picker_type}")


def main() -> None:
    typer.run(proc_pick_peaks)


if __name__ == "__main__":
    main()
