from __future__ import annotations

import cyclopts
import hashlib
import joblib
import polars as pl
import shutil
import subprocess
import yaml
import zipfile
from bfabric import Bfabric
from bfabric.entities import Storage
from dataclasses import dataclass
from loguru import logger
from pathlib import Path


from depiction_targeted_preproc.app.workunit_config import WorkunitConfig
from depiction_targeted_preproc.pipeline.setup import write_standardized_table
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import (
    get_result_files,
)
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke
from depiction_targeted_preprocbatch.batch_dataset import BatchDataset


@dataclass
class BatchJob:
    """Defines one job to be executed, including all required information."""

    imzml_relative_path: Path
    imzml_storage: Storage
    imzml_checksum: str
    panel_df: pl.DataFrame
    pipeline_parameters: Path


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

        storage_ids = {job.imzml.data_dict["storage"]["id"] for job in batch_dataset.jobs}
        storages = Storage.find_all(ids=sorted(storage_ids), client=self._client)
        jobs = [
            BatchJob(
                imzml_relative_path=Path(job.imzml.data_dict["relativepath"]),
                imzml_storage=storages[job.imzml.data_dict["storage"]["id"]],
                imzml_checksum=job.imzml.data_dict["filechecksum"],
                panel_df=job.panel.to_polars(),
                pipeline_parameters=pipeline_parameters,
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
        self._stage_inputs(job=job, sample_dir=sample_dir)

        # invoke the pipeline
        result_files = self._determine_result_files(job_dir=job_dir)
        SnakemakeInvoke().invoke(work_dir=job_dir, result_files=result_files)

        # export the results
        # TODO do not hardcode id
        output_storage = Storage.find(id=2, client=self._client)
        self._export_results(sample_name=sample_dir.name, result_files=result_files, output_storage=output_storage)

    def _stage_inputs(self, job: BatchJob, sample_dir: Path) -> None:
        """Stages all required input files for a particular job."""
        self._stage_imzml(
            relative_path=job.imzml_relative_path,
            input_storage=job.imzml_storage,
            sample_dir=sample_dir,
            checksum=job.imzml_checksum,
        )
        self._stage_panel(
            sample_dir=sample_dir,
            panel_df=job.panel_df,
        )
        self._stage_pipeline_parameters(sample_dir=sample_dir)

    def _export_results(self, sample_name: str, result_files: list[Path], output_storage: Storage) -> None:
        """Exports the results of one job."""
        # TODO export result as a zip file for the particular sample
        # create the zip file
        self.output_dir.mkdir(exist_ok=True, parents=True)
        zip_file_path = self.output_dir / f"{sample_name}.zip"
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for result_file in result_files:
                zip_file.write(result_file, arcname=Path(sample_name) / result_file.name)

        # Copy the zip file
        output_path = self._workunit_config.output_folder_absolute_path / zip_file_path.name
        output_path_relative = output_path.relative_to(output_storage.base_path)
        output_uri = f"{output_storage.scp_prefix}{output_path_relative}"
        self._scp(zip_file_path, output_uri)

        # Register the zip file in the workunit
        checksum = self._md5sum(zip_file_path)
        self._client.save(
            "resource",
            {
                "name": zip_file_path.name,
                "workunitid": self._workunit_config.workunit_id,
                "storageid": output_storage.id,
                "relativepath": output_path_relative,
                "filechecksum": checksum,
                "status": "available",
                "size": zip_file_path.stat().st_size,
            },
        )

    def _determine_result_files(self, job_dir: Path) -> list[Path]:
        """Returns the requested result files based on the pipeline parameters for a particular job."""
        pipeline_params = PipelineParameters.parse_yaml(path=job_dir / job_dir.stem / "pipeline_params.yml")
        return get_result_files(params=pipeline_params, work_dir=job_dir, sample_name=job_dir.stem)

    def _stage_imzml(self, relative_path: Path, input_storage: Storage, sample_dir: Path, checksum: str) -> None:
        """Copies the `raw.imzML` and `raw.ibd` files to the sample directory.
        This method assumes the position will be on a remote server, and first needs to be copied with scp.
        :param relative_path: Relative path of the imzML file (relative to storage roo-).
        :param input_storage: Storage of the imzML file.
        :param sample_dir: Directory to copy the files to.
        :param checksum: Expected checksum of the imzML file.
        """
        # Check for some not-yet supported functionality (TODO)
        if relative_path.suffix != ".imzML":
            # TODO implement this later
            raise NotImplementedError(
                "Currently only .imzML files are supported, .imzML.zip will be supported in the future"
            )

        # determine the paths to copy from
        input_paths = [relative_path, relative_path.with_suffix(".ibd")]
        scp_uris = [f"{input_storage.scp_prefix}{path}" for path in input_paths]

        # perform the copies
        for scp_uri, result_name in zip(scp_uris, ["raw.imzML", "raw.ibd"]):
            self._scp(scp_uri, str(sample_dir / result_name))

        # check the checksum
        actual_checksum = self._md5sum(sample_dir / "raw.imzML")
        if actual_checksum != checksum:
            raise ValueError(f"Checksum mismatch: expected {checksum}, got {actual_checksum}")

    def _stage_panel(self, sample_dir: Path, panel_df: pl.DataFrame) -> None:
        """Writes the marker panel to the sample directory."""
        write_standardized_table(input_df=panel_df, output_csv=sample_dir / "mass_list.raw.csv")

    def _stage_pipeline_parameters(self, sample_dir: Path) -> None:
        """Copies the `pipeline_params.yml` file to the particular sample's directory."""
        shutil.copyfile(
            sample_dir.parents[1] / "pipeline_params.yml",
            sample_dir / "pipeline_params.yml",
        )

    def _prepare_pipeline_parameters(self) -> Path:
        """Creates the `pipeline_params.yml` file in the work dir, which can then be copied for each sample.
        The path to the created file will be returned.
        """
        result_file = self._work_dir / "pipeline_params.yml"
        with result_file.open("w") as file:
            yaml.dump(self._workunit_config.pipeline_parameters.model_dump(mode="json"), file)
        return result_file

    def _md5sum(self, path: Path) -> str:
        """Returns the MD5 checksum of the file at the given path."""
        hasher = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(16384), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _scp(self, source: str | Path, target: str | Path) -> None:
        """Performs scp source target.
        Make sure that either the source or target specifies a host, otherwise you should just use shutil.copyfile.
        """
        logger.info(f"scp {source} {target}")
        subprocess.run(["scp", source, target], check=True)


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
