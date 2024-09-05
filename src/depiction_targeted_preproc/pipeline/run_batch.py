from pathlib import Path

import cyclopts
from bfabric import Bfabric
from bfabric.entities import Workunit
from depiction_targeted_preproc.pipeline.run import set_workunit_processing, set_workunit_available, run_one_job
from depiction_targeted_preprocbatch.batch_dataset import BatchDataset

app = cyclopts.App()


@app.default()
def run_batch(workunit_id: int, work_dir: Path, ssh_user: str | None = None, read_only: bool = False) -> None:
    client = Bfabric.from_config()
    if not read_only:
        set_workunit_processing(client=client, workunit_id=workunit_id)

    workunit = Workunit.find(id=workunit_id, client=client)
    batch_dataset = BatchDataset(dataset_id=workunit.input_dataset.id, client=client)

    # TODO there is currently a serious bug which prevents the parallelization here, but this would be the place to
    #      implement it
    for job in batch_dataset.jobs:
        run_one_job(
            client=client,
            work_dir=work_dir,
            sample_name=str(Path(job.imzml["name"]).stem),
            dataset_id=job.panel.id,
            workunit_id=workunit_id,
            imzml_resource_id=job.imzml.id,
            ssh_user=ssh_user,
            read_only=read_only,
        )

    if not read_only:
        set_workunit_available(client=client, workunit_id=workunit_id)


if __name__ == "__main__":
    app()
