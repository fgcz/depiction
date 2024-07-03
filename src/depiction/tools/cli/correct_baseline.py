from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
import yaml
from loguru import logger
from pydantic import BaseModel
from typer import Argument, Option

from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile
from depiction.tools.correct_baseline import BaselineVariants, CorrectBaseline

app = cyclopts.App()


class BaselineCorrectionConfig(BaseModel):
    n_jobs: int | None
    baseline_variant: BaselineVariants = BaselineVariants.TopHat
    window_size: int | float = 5000.0
    window_unit: Literal["ppm", "index"] = "ppm"


@app.command
def run_config(
    input_imzml: Annotated[Path, Argument()],
    output_imzml: Annotated[Path, Argument()],
    config: Annotated[Path, Argument()],
) -> None:
    parsed = BaselineCorrectionConfig.validate(yaml.safe_load(config.read_text()))
    correct_baseline(config=parsed, input_imzml=input_imzml, output_imzml=output_imzml)


@app.default
def run(
    input_imzml: Annotated[Path, Argument()],
    output_imzml: Annotated[Path, Argument()],
    n_jobs: Annotated[int, Option()] = None,
    baseline_variant: Annotated[BaselineVariants, Option()] = BaselineVariants.TopHat,
    window_size: Annotated[int | float, Option()] = 5000,
    window_unit: Annotated[Literal["ppm", "index"], Option()] = "ppm",
):
    config = BaselineCorrectionConfig.validate(
        dict(n_jobs=n_jobs, baseline_variant=baseline_variant, window_size=window_size, window_unit=window_unit)
    )
    correct_baseline(config=config, input_imzml=input_imzml, output_imzml=output_imzml)


def correct_baseline(config: BaselineCorrectionConfig, input_imzml: Path, output_imzml: Path) -> None:
    """Removes the baseline from the input imzML file and writes the result to the output imzML file."""
    output_imzml.parent.mkdir(parents=True, exist_ok=True)
    if config.baseline_variant == BaselineVariants.Zero:
        logger.info("Baseline correction is deactivated, copying input to output")
        shutil.copyfile(input_imzml, output_imzml)
        shutil.copyfile(input_imzml.with_suffix(".ibd"), output_imzml.with_suffix(".ibd"))
    else:
        if config.n_jobs is None:
            # TODO define some sane default for None and -1 n_jobs e.g. use all available up to a limit (None) or use all (1-r)
            n_jobs = 10
        else:
            n_jobs = config.n_jobs
        parallel_config = ParallelConfig(n_jobs=n_jobs)
        input_file = ImzmlReadFile(input_imzml)
        output_file = ImzmlWriteFile(output_imzml, imzml_mode=input_file.imzml_mode)
        correct_baseline = CorrectBaseline.from_variant(
            parallel_config=parallel_config,
            variant=config.baseline_variant,
            window_size=config.window_size,
            window_unit=config.window_unit,
        )
        correct_baseline.evaluate_file(input_file, output_file)


if __name__ == "__main__":
    app()
