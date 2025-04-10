import cyclopts
import yaml
from pathlib import Path
from rich.pretty import pprint

from depiction_io.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlModeEnum
from depiction_tools.process_spectra.config import ProcessSpectraConfig
from depiction_tools.process_spectra.process import process_spectra

app = cyclopts.App()


@app.command
def validate(config_file: Path) -> ProcessSpectraConfig:
    config = ProcessSpectraConfig.model_validate(yaml.safe_load(config_file.read_text()))
    pprint(config)
    return config


@app.default
def run(input_imzml_file: Path, output_imzml_file: Path, config_file: Path) -> None:
    config = validate(config_file)
    read_file = ImzmlReadFile(input_imzml_file)
    # TODO in the future we might implement logic to determine when this can still be set to CONTINUOUS
    write_file = ImzmlWriteFile(output_imzml_file, ImzmlModeEnum.PROCESSED)
    process_spectra(read_file=read_file, write_file=write_file, config=config)


if __name__ == "__main__":
    app()
