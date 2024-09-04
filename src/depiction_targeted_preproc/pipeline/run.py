import zipfile
from pathlib import Path

import cyclopts
import yaml
from bfabric import Bfabric
from bfabric.entities import Workunit, Resource
from depiction_targeted_preproc.pipeline.prepare_inputs import prepare_inputs
from depiction_targeted_preproc.pipeline.prepare_params import prepare_params, Params
from depiction_targeted_preproc.pipeline.store_outputs import store_outputs
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import get_result_files_new
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke

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


def run_workflow(sample_dir: Path) -> Path:
    # TODO to be refactored
    params = Params.model_validate(yaml.safe_load((sample_dir / "params.yml").read_text()))
    result_files = get_result_files_new(requested_artifacts=params.requested_artifacts, sample_dir=sample_dir)

    # invoke snakemake
    # TODO note report file is deactivated because it's currently broken due to dependencies (jinja2)
    SnakemakeInvoke(report_file=None).invoke(work_dir=sample_dir.parent, result_files=result_files)

    # zip the results
    sample_name = sample_dir.name
    output_dir = sample_dir.parent / "output"
    output_dir.mkdir(exist_ok=True)
    zip_file_path = output_dir / f"{sample_name}.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for result_file in result_files:
            zip_entry_path = result_file.relative_to(sample_dir.parent)
            zip_file.write(result_file, arcname=zip_entry_path)
    return zip_file_path


@app.default()
def run(workunit_id: int, work_dir: Path, ssh_user: str | None = None) -> None:
    client = Bfabric.from_config()
    dataset_id, imzml_resource_id, sample_name = _get_resource_flow_ids(client=client, workunit_id=workunit_id)
    sample_dir = work_dir / sample_name

    prepare_params(client=client, sample_dir=sample_dir, workunit_id=workunit_id)
    prepare_inputs(
        client=client,
        sample_dir=sample_dir,
        dataset_id=dataset_id,
        imzml_resource_id=imzml_resource_id,
        ssh_user=ssh_user,
    )
    _set_workunit_processing(client=client, workunit_id=workunit_id)
    zip_file_path = run_workflow(sample_dir=sample_dir)
    store_outputs(client=client, zip_file_path=zip_file_path, workunit_id=workunit_id, ssh_user=ssh_user)
    _set_workunit_available(client=client, workunit_id=workunit_id)


def _set_workunit_processing(client: Bfabric, workunit_id: int) -> None:
    """Sets the workunit to processing and deletes the default resource if it is available."""
    client.save("workunit", {"id": workunit_id, "status": "processing"})
    # TODO the default resource should be deleted in the future, but right now we simply set it to 0 and available
    #      (it will not be possible before updated wrapper creator)
    resources = Resource.find_by({"name": "% - resource", "workunitid": workunit_id}, client=client, max_results=1)
    if len(resources):
        client.save("resource", {"id": resources[0].id, "status": "available"})


def _set_workunit_available(client: Bfabric, workunit_id: int) -> None:
    client.save("workunit", {"id": workunit_id, "status": "available"})


if __name__ == "__main__":
    app()
