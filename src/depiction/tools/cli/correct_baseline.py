from __future__ import annotations

import shutil
from typing import Annotated, Literal, TYPE_CHECKING

import typer
from loguru import logger
from typer import Argument, Option

from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile
from depiction.tools.correct_baseline import BaselineVariants, CorrectBaseline

if TYPE_CHECKING:
    from pathlib import Path


def correct_baseline(
    input_imzml: Annotated[Path, Argument()],
    output_imzml: Annotated[Path, Argument()],
    n_jobs: Annotated[int, Option()] = None,
    baseline_variant: Annotated[BaselineVariants, Option()] = BaselineVariants.TopHat,
    window_size: Annotated[int | float, Option()] = 5000,
    window_unit: Annotated[Literal["ppm", "index"], Option()] = "ppm",
) -> None:
    """Removes the baseline from the input imzML file and writes the result to the output imzML file."""
    output_imzml.parent.mkdir(parents=True, exist_ok=True)
    if baseline_variant == BaselineVariants.Zero:
        logger.info("Baseline correction is deactivated, copying input to output")
        shutil.copyfile(input_imzml, output_imzml)
        shutil.copyfile(input_imzml.with_suffix(".ibd"), output_imzml.with_suffix(".ibd"))
    else:
        if n_jobs is None:
            # TODO define some sane default for None and -1 n_jobs e.g. use all available up to a limit (None) or use all (1-r)
            n_jobs = 10
        parallel_config = ParallelConfig(n_jobs=n_jobs)
        input_file = ImzmlReadFile(input_imzml)
        output_file = ImzmlWriteFile(output_imzml, imzml_mode=input_file.imzml_mode)
        correct_baseline = CorrectBaseline.from_variant(
            parallel_config=parallel_config, variant=baseline_variant, window_size=window_size, window_unit=window_unit
        )
        correct_baseline.evaluate_file(input_file, output_file)


if __name__ == "__main__":
    typer.run(correct_baseline)
