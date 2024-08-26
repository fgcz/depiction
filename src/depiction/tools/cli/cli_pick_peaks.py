from __future__ import annotations

from pathlib import Path

import cyclopts
import yaml

from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlModeEnum
from depiction.tools.pick_peaks import (
    PickPeaksConfig,
    pick_peaks,
    PeakPickerFindMFPyConfig,
    PeakPickerMSPeakPickerConfig,
)

app = cyclopts.App()


@app.command
def run_config(
    config: Path,
    input_imzml: Path,
    output_imzml: Path,
) -> None:
    """Runs the configured peak picker on input imzml file and writes the output to output imzml file."""
    config = PickPeaksConfig.model_validate(yaml.safe_load(config.read_text()))
    pick_peaks(
        config=config,
        input_file=ImzmlReadFile(input_imzml),
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED),
    )


@app.command
def run_findmf(
    input_imzml: Path,
    output_imzml: Path,
    *,
    n_jobs: int | None = None,
    resolution: float = 10000.0,
) -> None:
    """Runs FindMF peak picker on input imzml file and writes the output to output imzml file."""
    picker_config = PeakPickerFindMFPyConfig(resolution=resolution)
    pick_peaks(
        config=PickPeaksConfig(peak_picker=picker_config, peak_filtering=None, n_jobs=n_jobs),
        input_file=ImzmlReadFile(input_imzml),
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED),
    )


@app.command()
def run_mspeak(
    input_imzml: Path,
    output_imzml: Path,
    *,
    n_jobs: int | None = None,
    fit_type: str = "quadratic",
) -> None:
    """Runs MSPeakPicker on input imzml file and writes the output to output imzml file."""
    pick_peaks(
        config=PickPeaksConfig(
            peak_picker=PeakPickerMSPeakPickerConfig(fit_type=fit_type), peak_filtering=None, n_jobs=n_jobs
        ),
        input_file=ImzmlReadFile(input_imzml),
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED),
    )


if __name__ == "__main__":
    app()
