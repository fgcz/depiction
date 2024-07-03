from pathlib import Path
from typing import Annotated

import typer

from depiction.tools.cli.correct_baseline import correct_baseline
from depiction.tools.correct_baseline import BaselineVariants
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters, BaselineAdjustmentTophat


def proc_correct_baseline(
    input_imzml_path: Annotated[Path, typer.Option()],
    config_path: Annotated[Path, typer.Option()],
    output_imzml_path: Annotated[Path, typer.Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    window = {}
    match config.baseline_adjustment:
        case None:
            baseline_variant = BaselineVariants.Zero
        case BaselineAdjustmentTophat(window_size=window_size, window_unit=window_unit):
            baseline_variant = BaselineVariants.TopHat
            window["window_size"] = window_size
            window["window_unit"] = window_unit
        case _:
            raise ValueError(f"Unsupported baseline adjustment type: {config.baseline_adjustment.baseline_variant}")

    correct_baseline(
        input_imzml=input_imzml_path,
        output_imzml=output_imzml_path,
        n_jobs=config.n_jobs,
        baseline_variant=baseline_variant,
        **window,
    )


def main() -> None:
    typer.run(proc_correct_baseline)


if __name__ == "__main__":
    main()
