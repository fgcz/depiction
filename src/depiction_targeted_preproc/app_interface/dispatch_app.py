from __future__ import annotations

from pathlib import Path

import cyclopts
import yaml

from bfabric import Bfabric
from bfabric.entities import Resource
from bfabric.experimental.app_interface.workunit.definition import WorkunitDefinition
from depiction_targeted_preproc.pipeline.prepare_inputs import write_inputs_spec
from depiction_targeted_preproc.pipeline.prepare_params import parse_params
from depiction_targeted_preprocbatch.batch_dataset import BatchDataset

app = cyclopts.App()


@app.default
def dispatch_app(workunit_ref: int | Path, work_dir: Path) -> None:
    client = Bfabric.from_config()
    work_dir.mkdir(exist_ok=True, parents=True)
    workunit_definition = WorkunitDefinition.from_ref(workunit_ref, client)
    workunit_definition.to_yaml(work_dir / "workunit_definition.yml")

    params = parse_params(workunit_definition.execution)
    chunks = []

    if workunit_definition.execution.resources:
        # resource flow
        dataset_id = int(params["mass_list_id"])
        input_resources = Resource.find_all(ids=workunit_definition.execution.resources, client=client).values()
        imzml_resource_ids = [resource.id for resource in input_resources if resource["name"].endswith(".imzML")]

        # dispatch each input
        for imzml_resource_id in imzml_resource_ids:
            chunk_dir = work_dir / str(imzml_resource_id)
            write_inputs_spec(
                dataset_id=dataset_id, imzml_resource_id=imzml_resource_id, client=client, sample_dir=chunk_dir
            )
            write_params(params_dict=params, file=chunk_dir / "params.yml")
            chunks.append(chunk_dir)
    elif workunit_definition.execution.dataset:
        batch_dataset = BatchDataset(dataset_id=workunit_definition.execution.dataset, client=client)
        for job in batch_dataset.jobs:
            chunk_dir = work_dir / str(job.imzml.id)
            write_inputs_spec(
                dataset_id=job.panel.id, imzml_resource_id=job.imzml.id, client=client, sample_dir=chunk_dir
            )
            write_params(params_dict=params, file=chunk_dir / "params.yml")
            chunks.append(chunk_dir)

    # TODO consider how to best handle this
    with (work_dir / "chunks.yml").open("w") as f:
        data = {"chunks": [str(chunk) for chunk in chunks]}
        yaml.safe_dump(data, f)


def write_params(params_dict: dict, file: Path) -> None:
    with file.open("w") as f:
        yaml.safe_dump(params_dict, f)


if __name__ == "__main__":
    app()
