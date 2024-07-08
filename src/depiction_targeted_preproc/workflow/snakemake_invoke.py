import contextlib
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

    def invoke(
        self, work_dir: Path, result_files: list[Path], *, n_cores: int = 1, env_variables: dict[str, str] | None = None
    ) -> None:
        """Invokes the snakemake workflow to generate the requested result files.
        :param work_dir: The working directory where the data folder structure is located.
        :param result_files: The list of result files to generate (relative to `work_dir`).
        :param n_cores: The number of cores to use for the workflow execution.
        :param env_variables: Environment variables to set for the workflow execution.
        """
        env_variables = env_variables or {}
        if self.use_subprocess:
            self._invoke_subprocess(work_dir, result_files, n_cores=n_cores, env_variables=env_variables)
        else:
            self._invoke_direct(work_dir, result_files, n_cores=n_cores, env_variables=env_variables)

    @property
    def snakefile_path(self) -> Path:
        return Path(__file__).parents[1] / "workflow" / self.snakefile_name

    @property
    def workflow_dir(self) -> Path:
        return Path(__file__).parents[1] / "workflow"

    def _invoke_direct(
        self, work_dir: Path, result_files: list[Path], n_cores: int, env_variables: dict[str, str]
    ) -> None:
        from snakemake.api import SnakemakeApi
        from snakemake.settings import OutputSettings, StorageSettings, ResourceSettings, DAGSettings, ExecutionSettings

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
            with self._set_env_vars(env_variables):
                dag_api.execute_workflow(execution_settings=ExecutionSettings(keep_going=self.continue_on_error))

    def _invoke_subprocess(
        self, work_dir: Path, result_files: list[Path], n_cores: int, env_variables: dict[str, str]
    ) -> None:
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
            env={**os.environ, **env_variables},
        )
        if self.report_file:
            command = [
                *base_command,
                "--report",
                self.report_file,
                *[str(file.relative_to(work_dir)) for file in result_files],
            ]
            logger.info("Executing {command}", command=command)
            subprocess.run(
                command,
                cwd=self.workflow_dir,
                check=True,
                env={**os.environ, **env_variables},
            )

    @contextlib.contextmanager
    def _set_env_vars(self, env_variables: dict[str, str]) -> None:
        """Temporarily sets the given environment variables for the duration of the context."""
        old_env = os.environ.copy()
        os.environ.update(env_variables)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old_env)
