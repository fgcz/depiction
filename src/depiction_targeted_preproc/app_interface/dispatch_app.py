from __future__ import annotations

import cyclopts
import yaml
from pathlib import Path
from typing import Any

from app_runner.dispatch.dispatch_individual_resources import DispatchIndividualResources, config_msi_imzml
from bfabric import Bfabric
from bfabric.entities import Resource
from bfabric.experimental.workunit_definition import WorkunitDefinition
from depiction_targeted_preproc.pipeline.prepare_inputs import write_inputs_spec
from depiction_targeted_preproc.pipeline.prepare_params import parse_params

app = cyclopts.App()


class DispatchApp(DispatchIndividualResources):
    # TODO rename params to raw_params?
    def dispatch_job(self, resource: Resource, params: dict[str, Any]) -> Path:
        params_parsed = parse_params(params)
        chunk_dir = self._out_dir / Path(resource["name"]).stem
        chunk_dir.mkdir(exist_ok=True, parents=True)
        write_inputs_spec(
            dataset_id=params["mass_list_id"],
            imzml_resource_id=resource.id,
            client=self._client,
            sample_dir=chunk_dir,
        )
        write_params(params_dict=params_parsed, file=chunk_dir / "params.yml")
        return chunk_dir


@app.default
def dispatch_app(workunit_ref: int | Path, work_dir: Path) -> None:
    client = Bfabric.from_config()
    workunit_definition = WorkunitDefinition.from_ref(
        workunit_ref, client, cache_file=work_dir / "workunit_definition.yml"
    )
    dispatcher = DispatchApp(client=client, config=config_msi_imzml(), out_dir=work_dir)
    dispatcher.dispatch_workunit(definition=workunit_definition)


def write_params(params_dict: dict, file: Path) -> None:
    with file.open("w") as f:
        yaml.safe_dump(params_dict, f)


if __name__ == "__main__":
    app()
