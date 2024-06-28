import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class SnakemakeInvoke:
    snakefile_name: str = "Snakefile"
    use_subprocess: bool = True
    continue_on_error: bool = False
    report_file: str | None = "report.html"

    def invoke(self, work_dir: Path, result_files: list[Path], n_cores: int = 1) -> None:
        if self.use_subprocess:
            self._invoke_subprocess(work_dir, result_files, n_cores)
        else:
            self._invoke_direct(work_dir, result_files, n_cores)

    @property
    def snakefile_path(self) -> Path:
        return Path(__file__).parents[1] / "workflow" / self.snakefile_name

    @property
    def workflow_dir(self) -> Path:
        return Path(__file__).parents[1] / "workflow"

    def _invoke_direct(self, work_dir: Path, result_files: list[Path], n_cores: int) -> None:
        from snakemake.api import SnakemakeApi
        from snakemake.settings import OutputSettings
        from snakemake.settings import StorageSettings
        from snakemake.settings import ResourceSettings
        from snakemake.settings import DAGSettings
        from snakemake.settings import ExecutionSettings

        with SnakemakeApi(
            OutputSettings(
                verbose=True,
                show_failed_logs=True,
            ),
        ) as snakemake_api:
            workflow_api = snakemake_api.workflow(
                storage_settings=StorageSettings(),
                resource_settings=ResourceSettings(cores=n_cores),
                snakefile=self.snakefile_path,
                workdir=work_dir,
            )
            dag_api = workflow_api.dag(
                dag_settings=DAGSettings(targets=[str(p) for p in result_files], force_incomplete=True)
            )
            dag_api.execute_workflow(execution_settings=ExecutionSettings(keep_going=self.continue_on_error))

    def _invoke_subprocess(self, work_dir: Path, result_files: list[Path], n_cores: int) -> None:
        snakemake_bin = shutil.which("snakemake")
        if snakemake_bin is None:
            raise RuntimeError(f"snakemake not found, check PATH: {os.environ['PATH']}")
        extra_args = []
        if self.continue_on_error:
            extra_args.append("--keep-going")
        # if self.report_file:
        #    extra_args.extend(["--report", self.report_file])
        base_command = [
            snakemake_bin,
            "-d",
            str(work_dir),
            "--cores",
            str(n_cores),
            "--snakefile",
            str(self.snakefile_path),
            # TODO configurable
            "--rerun-incomplete",
            *extra_args,
        ]
        command = [
            *base_command,
            *[str(file.relative_to(work_dir)) for file in result_files],
        ]
        logger.info("Executing {command}", command=command)
        subprocess.run(
            command,
            cwd=self.workflow_dir,
            check=True,
        )
        if self.report_file:
            command = [
                *base_command,
                "--report",
                self.report_file,
            ]
            logger.info("Executing {command}", command=command)
            subprocess.run(
                command,
                cwd=self.workflow_dir,
                check=True,
            )
