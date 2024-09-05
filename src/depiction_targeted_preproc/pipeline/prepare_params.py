from pathlib import Path

import cyclopts
import yaml
from bfabric import Bfabric
from bfabric.entities import Workunit
from pydantic import BaseModel

from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact


class Params(BaseModel):
    config_preset: str
    requested_artifacts: list[PipelineArtifact]
    n_jobs: int = 32


def _get_params(client: Bfabric, workunit_id: int) -> dict[str, str | int | bool]:
    workunit = Workunit.find(id=workunit_id, client=client)
    requested_artifacts = []
    if workunit.parameter_values["output_activate_calibrated_imzml"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMZML)
    if workunit.parameter_values["output_activate_calibrated_ometiff"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_IMAGES)
    if workunit.parameter_values["output_activate_calibration_qc"] == "true":
        requested_artifacts.append(PipelineArtifact.CALIB_QC)
    return Params(
        config_preset=workunit.parameter_values["config_preset"], requested_artifacts=requested_artifacts
    ).model_dump(mode="json")


def prepare_params(
    client: Bfabric,
    sample_dir: Path,
    workunit_id: int,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    params_yaml = sample_dir / "params.yml"
    with params_yaml.open("w") as file:
        yaml.safe_dump(_get_params(client=client, workunit_id=workunit_id), file)


app = cyclopts.App()


@app.default
def prepare_params_from_cli(
    sample_dir: Path,
    config_preset: str,
    requested_artifacts: list[str],
    n_jobs: int = 32,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    params_yaml = sample_dir / "params.yml"
    with params_yaml.open("w") as file:
        yaml.safe_dump(
            Params(config_preset=config_preset, requested_artifacts=requested_artifacts, n_jobs=n_jobs).model_dump(
                mode="json"
            ),
            file,
        )


if __name__ == "__main__":
    app()
