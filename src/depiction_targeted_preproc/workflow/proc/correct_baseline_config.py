# TODO this should be handled better in the future, but for illustrative purposes I'm doing it here
from pathlib import Path

import cyclopts
import yaml

from depiction_targeted_preproc.pipeline_config.model import PipelineParameters

app = cyclopts.App()


@app.default
def correct_baseline_config(input_config: Path, output_config: Path) -> None:
    config = PipelineParameters.parse_yaml(input_config)
    with output_config.open("w") as file:
        yaml.dump(config.baseline_correction.dict(), file)


if __name__ == "__main__":
    app()
