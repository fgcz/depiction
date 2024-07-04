from __future__ import annotations

from pathlib import Path

import cyclopts
import yaml

from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlModeEnum
from depiction.tools.calibrate import CalibrationConfig, calibrate

app = cyclopts.App()


@app.command
def run_config(
    config: Path,
    input_imzml: Path,
    output_imzml: Path,
    *,
    input_mass_list: Path | None = None,
    output_calib_data: Path | None = None,
) -> None:
    parsed = CalibrationConfig.validate(yaml.safe_load(config.read_text()))
    input_file = ImzmlReadFile(input_imzml)
    output_file = ImzmlWriteFile(output_imzml, imzml_mode=ImzmlModeEnum.PROCESSED)
    calibrate(
        config=parsed,
        input_file=input_file,
        output_file=output_file,
        mass_list=input_mass_list,
        coefficient_output_path=output_calib_data,
    )


if __name__ == "__main__":
    app()
