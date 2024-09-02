from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import cyclopts
from bfabric import Bfabric
from bfabric.entities import Storage, Resource
from depiction_targeted_preproc.app.workunit_config import WorkunitConfig
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import (
    get_result_files,
)
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke
from depiction_targeted_preprocbatch.batch_dataset import BatchDataset
from depiction_targeted_preprocbatch.job_export_results import JobExportResults
from depiction_targeted_preprocbatch.job_prepare_inputs import JobPrepareInputs
from loguru import logger


@dataclass
class BatchJob:
    """Defines one job to be executed, including all required information."""

    imzml_resource_id: int
    pipeline_parameters: PipelineParameters
    dataset_id: int
    sample_name: str
    ssh_user: str | None = None

    @classmethod
    def from_bfabric(cls, imzml_resource: Resource, workunit_config: WorkunitConfig, ssh_user: str | None) -> BatchJob:
        return cls(
            imzml_resource_id=imzml_resource.id,
            pipeline_parameters=workunit_config.pipeline_parameters,
            dataset_id=workunit_config.input_dataset_id,
            sample_name=Path(imzml_resource["name"]).stem,
            ssh_user=ssh_user,
        )


class Executor:
    """Executes the pipeline for multiple files, supporting parallel, but isolated, execution of multiple jobs.
    Results will be gradually published once they become available.
    """

    def __init__(
        self,
        proc_dir: Path,
        output_dir: Path,
        workunit_config: WorkunitConfig,
        client: Bfabric,
        force_ssh_user: str | None = None,
    ) -> None:
        self._client = client
        self._workunit_config = workunit_config
        self._force_ssh_user = force_ssh_user
        self.proc_dir = proc_dir
        self.output_dir = output_dir

    def run(self, n_jobs: int) -> None:
        """Runs all jobs, executing up to `n_jobs` in parallel."""
        self._set_workunit_processing()
        batch_dataset = BatchDataset(dataset_id=self._workunit_config.input_dataset_id, client=self._client)

        jobs = [
            BatchJob.from_bfabric(
                imzml_resource=job.imzml, workunit_config=self._workunit_config, ssh_user=self._force_ssh_user
            )
            for job in batch_dataset.jobs
        ]
        # TODO parallelization is currently broken
        if n_jobs != 1:
            logger.error("Parallelization is currently broken and will be disabled.")
        for job in jobs:
            self.run_job(job)
        # parallel = joblib.Parallel(n_jobs=n_jobs, verbose=10)
        # parallel(joblib.delayed(self.run_job)(job) for job in jobs)
        self._set_workunit_available()

    def run_job(self, job: BatchJob) -> None:
        """Runs a single job."""
        workflow_dir = self.proc_dir / job.sample_name
        sample_dir = workflow_dir / job.sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # stage input
        logger.debug(f"Preparing inputs for {job}")
        JobPrepareInputs.prepare(job=job, sample_dir=sample_dir, client=self._client)

        # invoke the pipeline
        logger.debug(f"Running pipeline for {job}")
        result_files = self._determine_result_files(job=job, workflow_dir=workflow_dir)
        SnakemakeInvoke().invoke(work_dir=workflow_dir, result_files=result_files)

        # export the results
        logger.debug(f"Exporting results for {job}")
        # TODO do not hardcode id
        output_storage = Storage.find(id=2, client=self._client)
        JobExportResults.export(
            client=self._client,
            work_dir=workflow_dir,
            workunit_config=self._workunit_config,
            sample_name=sample_dir.name,
            result_files=result_files,
            output_storage=output_storage,
            force_ssh_user=self._force_ssh_user,
        )

    def _determine_result_files(self, job: BatchJob, workflow_dir: Path) -> list[Path]:
        """Returns the requested result files based on the pipeline parameters for a particular job."""
        return get_result_files(params=job.pipeline_parameters, work_dir=workflow_dir, sample_name=job.sample_name)

    def _set_workunit_processing(self) -> None:
        """Sets the workunit to processing and deletes the default resource if it is available."""
        self._client.save(
            "workunit",
            {
                "id": self._workunit_config.workunit_id,
                "status": "processing",
            },
        )
        JobExportResults.delete_default_resource(workunit_id=self._workunit_config.workunit_id, client=self._client)

    def _set_workunit_available(self) -> None:
        # TODO not clear if it needs to be addressed here or in the shell script
        self._client.save(
            "workunit",
            {
                "id": self._workunit_config.workunit_id,
                "status": "available",
            },
        )


app = cyclopts.App()


@app.default
def process_app(
    proc_dir: Path,
    output_dir: Path,
    workunit_yaml: Path,
    n_jobs: int = 32,
    force_ssh_user: str | None = None,
) -> None:
    """Runs the executor."""
    workunit_config = WorkunitConfig.from_yaml(workunit_yaml)
    client = Bfabric.from_config()
    executor = Executor(
        proc_dir=proc_dir,
        output_dir=output_dir,
        workunit_config=workunit_config,
        client=client,
        force_ssh_user=force_ssh_user,
    )
    executor.run(n_jobs=n_jobs)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app()
