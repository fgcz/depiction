from __future__ import annotations

import cyclopts
import yaml
from pathlib import Path

from depiction_targeted_preproc.pipeline_config.model import PipelineParameters

app = cyclopts.App()


@app.default
def process_spectra_config(input_config: Path, output_config: Path) -> None:
    config = PipelineParameters.parse_yaml(input_config)
    with output_config.open("w") as file:
        yaml.dump(config.process_spectra.model_dump(mode="json"), file)


if __name__ == "__main__":
    app()
