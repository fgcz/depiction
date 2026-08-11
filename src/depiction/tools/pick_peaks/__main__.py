from __future__ import annotations

import cyclopts
import yaml
from loguru import logger
from pathlib import Path

from depiction_io import ImzmlModeEnum, ImzmlWriteFile, get_read_file
from depiction.tools.pick_peaks.config import PickPeaksConfig, PeakPickerFindMFPyConfig, PeakPickerMSPeakPickerConfig
from depiction.tools.pick_peaks.pick_peaks import pick_peaks

app = cyclopts.App()


@app.command
def run_config(
    config: Path,
    input_imzml: Path,
    output_imzml: Path,
) -> None:
    """Runs the configured peak picker on input imzml file and writes the output to output imzml file."""
    raw_config = yaml.safe_load(config.read_text())
    if raw_config is None:
        logger.info("Peak picking deactivated, copying input to output.")
        get_read_file(input_imzml).copy_to(output_imzml)
    else:
        config = PickPeaksConfig.model_validate(raw_config)
        input_file = get_read_file(input_imzml)
        pick_peaks(
            config=config,
            input_file=input_file,
            output_file=ImzmlWriteFile(
                output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED, pixel_size=input_file.pixel_size
            ),
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
    input_file = get_read_file(input_imzml)
    pick_peaks(
        config=PickPeaksConfig(peak_picker=picker_config, peak_filtering=None, n_jobs=n_jobs),
        input_file=input_file,
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED, pixel_size=input_file.pixel_size),
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
    input_file = get_read_file(input_imzml)
    pick_peaks(
        config=PickPeaksConfig(
            peak_picker=PeakPickerMSPeakPickerConfig(fit_type=fit_type), peak_filtering=None, n_jobs=n_jobs
        ),
        input_file=input_file,
        output_file=ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED, pixel_size=input_file.pixel_size),
    )


if __name__ == "__main__":
    app()
