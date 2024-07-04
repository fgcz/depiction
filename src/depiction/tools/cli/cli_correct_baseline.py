from __future__ import annotations

from pathlib import Path
from typing import Literal

import cyclopts
import yaml

from depiction.tools.correct_baseline import BaselineVariants, BaselineCorrectionConfig, correct_baseline

app = cyclopts.App()


@app.command
def run_config(
    config: Path,
    input_imzml: Path,
    output_imzml: Path,
) -> None:
    parsed = BaselineCorrectionConfig.validate(yaml.safe_load(config.read_text()))
    correct_baseline(config=parsed, input_imzml=input_imzml, output_imzml=output_imzml)


@app.default
def run(
    input_imzml: Path,
    output_imzml: Path,
    *,
    n_jobs: int | None = None,
    baseline_variant: BaselineVariants = BaselineVariants.TopHat,
    window_size: int | float = 5000,
    window_unit: Literal["ppm", "index"] = "ppm",
) -> None:
    config = BaselineCorrectionConfig(
        n_jobs=n_jobs, baseline_variant=baseline_variant, window_size=window_size, window_unit=window_unit
    )
    correct_baseline(config=config, input_imzml=input_imzml, output_imzml=output_imzml)


if __name__ == "__main__":
    app()
