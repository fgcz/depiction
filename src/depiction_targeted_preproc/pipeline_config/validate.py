from pathlib import Path

import cyclopts
import yaml
from rich.pretty import pprint

from depiction_targeted_preproc.pipeline_config.model import PipelineParametersPreset, PipelineParameters

app = cyclopts.App()


@app.command
def preset(yaml_path: Path) -> None:
    data = yaml.safe_load(yaml_path.read_text())
    parsed = PipelineParametersPreset.model_validate(data)
    pprint(parsed)


@app.command
def params(yaml_path: Path) -> None:
    data = yaml.safe_load(yaml_path.read_text())
    parsed = PipelineParameters.model_validate(data)
    pprint(parsed)


if __name__ == "__main__":
    app()
