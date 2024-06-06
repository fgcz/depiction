import shutil
from pathlib import Path
from typing import Annotated
import typer
from loguru import logger

from depiction.spectrum.baseline.tophat_baseline import TophatBaseline
from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile
from depiction.tools.correct_baseline import CorrectBaseline

from depiction_targeted_preproc.pipeline_config.model import PipelineParameters, BaselineAdjustmentTophat


def correct_baseline(
    input_imzml_path: Annotated[Path, typer.Option()],
    config_path: Annotated[Path, typer.Option()],
    output_imzml_path: Annotated[Path, typer.Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    match config.baseline_adjustment:
        case None:
            logger.info("Baseline adjustment is deactivated")
            shutil.copy(input_imzml_path, output_imzml_path)
            shutil.copy(input_imzml_path.with_suffix(".ibd"), output_imzml_path.with_suffix(".ibd"))
        case BaselineAdjustmentTophat(window_size=window_size, window_unit=window_unit):
            baseline = TophatBaseline(window_size=window_size, window_unit=window_unit)
            parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)
            read_file = ImzmlReadFile(input_imzml_path)
            write_file = ImzmlWriteFile(output_imzml_path, imzml_mode=read_file.imzml_mode)
            correct_baseline = CorrectBaseline(parallel_config=parallel_config, baseline_correction=baseline)
            correct_baseline.evaluate_file(read_file, write_file)
        case _:
            raise ValueError(f"Unsupported baseline adjustment type: {config.baseline_adjustment.baseline_type}")


def main() -> None:
    typer.run(correct_baseline)


if __name__ == "__main__":
    main()
