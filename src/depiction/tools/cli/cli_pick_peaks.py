from __future__ import annotations

from pathlib import Path

import cyclopts
import yaml

from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlModeEnum
from depiction.tools.filter_peaks import FilterPeaksConfig, filter_peaks, FilterNHighestIntensityPartitionedConfig
from depiction.tools.pick_peaks import PickPeaksConfig, pick_peaks

app = cyclopts.App()


@app.command
def run_config(
    config: Path,
    input_imzml: Path,
    output_imzml: Path,
) -> None:
    config = PickPeaksConfig.validate(yaml.safe_load(config.read_text()))
    pick_peaks(
        config=config,
        input_file=ImzmlReadFile(input_imzml),
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED),
    )


# @app.default
# def run(
#    input_imzml: Path,
#    output_imzml: Path,
#    *,
#    n_jobs: int | None = None,
# ) -> None:
#    pass


if __name__ == "__main__":
    app()
