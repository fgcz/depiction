from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cyclopts
import joblib
import polars as pl
import yaml
from bfabric import Bfabric
from bfabric.entities import Storage

from depiction_targeted_preproc.app.workunit_config import WorkunitConfig
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import (
    get_result_files,
)
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke
from depiction_targeted_preprocbatch.batch_dataset import BatchDataset
from depiction_targeted_preprocbatch.job_export_results import JobExportResults
from depiction_targeted_preprocbatch.job_prepare_inputs import JobPrepareInputs


@dataclass
class BatchJob:
    """Defines one job to be executed, including all required information."""

    imzml_relative_path: Path
    imzml_storage: Storage
    imzml_checksum: str
    panel_df: pl.DataFrame
    pipeline_parameters: Path
    ssh_user: str | None = None


class Executor:
    """Executes the pipeline for multiple files, supporting parallel, but isolated, execution of multiple jobs.
    Results will be gradually published once they become available.
    """

    def __init__(
        self,
        work_dir: Path,
        workunit_config: WorkunitConfig,
        client: Bfabric,
        force_ssh_user: str | None = None,
    ) -> None:
        self._client = client
        self._work_dir = work_dir
        self._workunit_config = workunit_config
        self._force_ssh_user = force_ssh_user
        self.output_dir = work_dir / "output"

    def run(self, n_jobs: int) -> None:
        """Runs all jobs, executing up to `n_jobs` in parallel."""
        self._work_dir.mkdir(exist_ok=True, parents=True)
        batch_dataset = BatchDataset(dataset_id=self._workunit_config.input_dataset_id, client=self._client)
        pipeline_parameters = self._prepare_pipeline_parameters()

        storage_ids = {job.imzml["storage"]["id"] for job in batch_dataset.jobs}
        storages = Storage.find_all(ids=sorted(storage_ids), client=self._client)
        jobs = [
            BatchJob(
                imzml_relative_path=Path(job.imzml["relativepath"]),
                imzml_storage=storages[job.imzml["storage"]["id"]],
                imzml_checksum=job.imzml["filechecksum"],
                panel_df=job.panel.to_polars(),
                pipeline_parameters=pipeline_parameters,
                ssh_user=self._force_ssh_user,
            )
            for job in batch_dataset.jobs
        ]
        parallel = joblib.Parallel(n_jobs=n_jobs, verbose=10)
        parallel(joblib.delayed(self.run_job)(job) for job in jobs)

    def run_job(self, job: BatchJob) -> None:
        """Runs a single job."""
        job_dir = self._work_dir / job.imzml_relative_path.stem
        sample_dir = job_dir / job.imzml_relative_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        # stage input
        JobPrepareInputs.prepare(job=job, sample_dir=sample_dir)

        # invoke the pipeline
        result_files = self._determine_result_files(job_dir=job_dir)
        SnakemakeInvoke().invoke(work_dir=job_dir, result_files=result_files)

        # export the results
        # TODO do not hardcode id
        output_storage = Storage.find(id=2, client=self._client)
        JobExportResults.export(
            client=self._client,
            work_dir=self._work_dir,
            workunit_config=self._workunit_config,
            sample_name=sample_dir.name,
            result_files=result_files,
            output_storage=output_storage,
        )

    def _determine_result_files(self, job_dir: Path) -> list[Path]:
        """Returns the requested result files based on the pipeline parameters for a particular job."""
        pipeline_params = PipelineParameters.parse_yaml(path=job_dir / job_dir.stem / "pipeline_params.yml")
        return get_result_files(params=pipeline_params, work_dir=job_dir, sample_name=job_dir.stem)

    def _prepare_pipeline_parameters(self) -> Path:
        """Creates the `pipeline_params.yml` file in the work dir, which can then be copied for each sample.
        The path to the created file will be returned.
        """
        result_file = self._work_dir / "pipeline_params.yml"
        with result_file.open("w") as file:
            yaml.dump(self._workunit_config.pipeline_parameters.model_dump(mode="json"), file)
        return result_file


app = cyclopts.App()


@app.default
def process_app(
    work_dir: Path,
    workunit_yaml: Path,
    n_jobs: int = 32,
    force_ssh_user: str | None = None,
) -> None:
    """Runs the executor."""
    workunit_config = WorkunitConfig.from_yaml(workunit_yaml)
    client = Bfabric.from_config()
    executor = Executor(
        work_dir=work_dir, workunit_config=workunit_config, client=client, force_ssh_user=force_ssh_user
    )
    executor.run(n_jobs=n_jobs)


if __name__ == "__main__":
    app()
