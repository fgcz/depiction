import cyclopts
import yaml
from pathlib import Path

from depiction.tools.calibrate.config import (
    CalibrationConfig,
    CalibrationRegressShiftConfig,
    CalibrationChemicalPeptideNoiseConfig,
    CalibrationMCCConfig,
    CalibrationConstantGlobalShiftConfig,
)

app = cyclopts.App()


def remove_smoothing(config: CalibrationConfig) -> CalibrationConfig:
    match config:
        case CalibrationRegressShiftConfig():
            return config.model_copy(update={"spatial_smoothing": None})
        case CalibrationChemicalPeptideNoiseConfig():
            return config
        case CalibrationMCCConfig():
            return config.model_copy(update={"coef_smoothing_activated": False})
        case CalibrationConstantGlobalShiftConfig():
            return config
        case _:
            raise NotImplementedError(f"Smoothing removal for {config.method.calibration_method} not implemented.")


@app.default
def prepare(input_config_path: Path, output_config_path: Path) -> None:
    config = CalibrationConfig.model_validate(yaml.safe_load(input_config_path.read_text()))
    config_new = remove_smoothing(config)
    output_config_path.write_text(yaml.safe_dump(config_new.model_dump(mode="json")))


if __name__ == "__main__":
    app()
