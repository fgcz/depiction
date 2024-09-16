from pathlib import Path

import cyclopts
from bfabric import Bfabric
from bfabric.entities import Workunit, Resource

from depiction_targeted_preproc.pipeline.prepare_inputs import prepare_inputs
from depiction_targeted_preproc.pipeline.prepare_params import prepare_params
from depiction_targeted_preproc.pipeline.run_workflow import run_workflow
from depiction_targeted_preproc.pipeline.store_outputs import store_outputs

app = cyclopts.App()


def _get_resource_flow_ids(client: Bfabric, workunit_id: int) -> tuple[int, int, str]:
    workunit = Workunit.find(id=workunit_id, client=client)
    dataset_id = workunit.parameter_values["mass_list_id"]
    imzml_resources = [r for r in workunit.input_resources if r["name"].endswith(".imzML")]
    if len(imzml_resources) != 1:
        raise ValueError(f"Expected exactly one .imzML resource, found {len(imzml_resources)}")
    imzml_resource_id = imzml_resources[0].id
    sample_name = Path(imzml_resources[0]["name"]).stem
    return dataset_id, imzml_resource_id, sample_name


def run_one_job(
    client: Bfabric,
    work_dir: Path,
    sample_name: str,
    dataset_id: int,
    workunit_id: int,
    imzml_resource_id: int,
    ssh_user: str | None,
    read_only: bool,
    prepare_only: bool = False,
    override_params: bool = False,
) -> None:
    sample_dir = work_dir / sample_name
    prepare_params(client=client, sample_dir=sample_dir, workunit_id=workunit_id, override=override_params)
    prepare_inputs(
        client=client,
        sample_dir=sample_dir,
        dataset_id=dataset_id,
        imzml_resource_id=imzml_resource_id,
        ssh_user=ssh_user,
    )
    if prepare_only:
        return
    zip_file_path = run_workflow(sample_dir=sample_dir)
    if not read_only:
        store_outputs(client=client, zip_file_path=zip_file_path, workunit_id=workunit_id, ssh_user=ssh_user)


@app.default()
def run_resource_flow(workunit_id: int, work_dir: Path, ssh_user: str | None = None, read_only: bool = False) -> None:
    client = Bfabric.from_config()
    if not read_only:
        set_workunit_processing(client=client, workunit_id=workunit_id)
    dataset_id, imzml_resource_id, sample_name = _get_resource_flow_ids(client=client, workunit_id=workunit_id)
    run_one_job(
        client=client,
        work_dir=work_dir,
        sample_name=sample_name,
        dataset_id=dataset_id,
        workunit_id=workunit_id,
        imzml_resource_id=imzml_resource_id,
        ssh_user=ssh_user,
        read_only=read_only,
    )
    if not read_only:
        set_workunit_available(client=client, workunit_id=workunit_id)


def set_workunit_processing(client: Bfabric, workunit_id: int) -> None:
    """Sets the workunit to processing and deletes the default resource if it is available."""
    client.save("workunit", {"id": workunit_id, "status": "processing"})
    # TODO the default resource should be deleted in the future, but right now we simply set it to 0 and available
    #      (it will not be possible before updated wrapper creator)
    resources = Resource.find_by({"name": "% - resource", "workunitid": workunit_id}, client=client, max_results=1)
    if resources:
        client.save("resource", {"id": list(resources.values())[0].id, "status": "available"})


def set_workunit_available(client: Bfabric, workunit_id: int) -> None:
    client.save("workunit", {"id": workunit_id, "status": "available"})


if __name__ == "__main__":
    app()
