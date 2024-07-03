from __future__ import annotations

from pathlib import Path

import cyclopts
import yaml

from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlModeEnum
from depiction.tools.filter_peaks import FilterPeaksConfig, filter_peaks, FilterNHighestIntensityPartitionedConfig

app = cyclopts.App()


@app.command
def run_config(
    config: Path,
    input_imzml: Path,
    output_imzml: Path,
) -> None:
    parsed = FilterPeaksConfig.validate(yaml.safe_load(config.read_text()))
    filter_peaks(
        config=parsed,
        input_file=ImzmlReadFile(input_imzml),
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED),
    )


@app.default
def run(
    input_imzml: Path,
    output_imzml: Path,
    *,
    n_jobs: int | None = None,
) -> None:
    # TODO this is hardcoded like before in the workflow
    peaks_filter = FilterNHighestIntensityPartitionedConfig(max_count=500, n_partitions=8)
    config = FilterPeaksConfig.validate(dict(filters=[peaks_filter], n_jobs=n_jobs))
    filter_peaks(
        config=config,
        input_file=ImzmlReadFile(input_imzml),
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED),
    )


if __name__ == "__main__":
    app()
