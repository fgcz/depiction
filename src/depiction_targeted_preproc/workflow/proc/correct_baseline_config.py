# TODO this should be handled better in the future, but for illustrative purposes I'm doing it here
from pathlib import Path

import cyclopts
import yaml

from depiction_targeted_preproc.pipeline_config.model import PipelineParameters, BaselineAdjustmentTophat

app = cyclopts.App()


@app.default
def correct_baseline_config(input_config: Path, output_config: Path) -> None:
    config = PipelineParameters.parse_yaml(input_config)
    args = {"n_jobs": config.n_jobs, "baseline_variant": config.baseline_adjustment.baseline_type or "Zero"}
    # TODO fix later
    if args["baseline_variant"] == "Tophat":
        args["baseline_variant"] = "TopHat"
    match config.baseline_adjustment:
        case BaselineAdjustmentTophat(window_size=window_size, window_unit=window_unit):
            args["window_size"] = window_size
            args["window_unit"] = window_unit
        case _:
            pass
    output_path = Path(output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.dump(args, f)


if __name__ == "__main__":
    app()
