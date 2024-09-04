import contextlib
import os
import shlex
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
    n_cores: int = 1
    env_variables: dict[str, str] | None = None

    def invoke(self, work_dir: Path, result_files: list[Path]) -> None:
        """Invokes the snakemake workflow to generate the requested result files.
        :param work_dir: The working directory where the data folder structure is located.
        :param result_files: The list of result files to generate (relative to `work_dir`).
        """
        if self.use_subprocess:
            self._invoke_subprocess(work_dir, result_files)
        else:
            self._invoke_direct(work_dir, result_files)

    def dry_run(self, work_dir: Path, result_files: list[Path]) -> None:
        self._invoke_subprocess(work_dir, result_files, extra_args=["--dryrun", "--printshellcmds"])

    @property
    def snakefile_path(self) -> Path:
        return Path(__file__).parents[1] / "workflow" / self.snakefile_name

    @property
    def workflow_dir(self) -> Path:
        return Path(__file__).parents[1] / "workflow"

    def _invoke_direct(self, work_dir: Path, result_files: list[Path]) -> None:
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
                resource_settings=ResourceSettings(cores=self.n_cores),
                snakefile=self.snakefile_path,
                workdir=work_dir,
            )
            dag_api = workflow_api.dag(
                dag_settings=DAGSettings(targets=[str(p) for p in result_files], force_incomplete=True)
            )
            with self._set_env_vars():
                dag_api.execute_workflow(execution_settings=ExecutionSettings(keep_going=self.continue_on_error))

    def _invoke_subprocess(self, work_dir: Path, result_files: list[Path], extra_args: list[str] | None = None) -> None:
        extra_args = extra_args or []
        if self.continue_on_error:
            extra_args.append("--keep-going")
        base_command = self.get_base_command(extra_args=extra_args, work_dir=work_dir)
        command = self.get_command_create_results(
            base_command=base_command, result_files=result_files, work_dir=work_dir
        )
        logger.info("Executing {command}", command=self._args_to_shell_command(command))
        subprocess.run(
            command,
            cwd=self.workflow_dir,
            check=True,
            env={**os.environ, **(self.env_variables or {})},
        )
        if self.report_file:
            command = self.get_command_create_report(
                base_command=base_command, result_files=result_files, work_dir=work_dir
            )
            logger.info("Executing {command}", command=command)
            subprocess.run(
                command,
                cwd=self.workflow_dir,
                check=True,
                env={**os.environ, **(self.env_variables or {})},
            )

    def get_base_command(self, extra_args: list[str], work_dir: Path) -> list[str]:
        snakemake_bin = shutil.which("snakemake")
        if snakemake_bin is None:
            raise RuntimeError(f"snakemake not found, check PATH: {os.environ['PATH']}")
        return [
            snakemake_bin,
            "-d",
            str(work_dir.absolute()),
            "--cores",
            str(self.n_cores),
            "--snakefile",
            str(self.snakefile_path),
            # TODO configurable
            "--rerun-incomplete",
            *extra_args,
        ]

    def get_command_create_results(
        self, base_command: list[str], result_files: list[Path], work_dir: Path
    ) -> list[str]:
        return [
            *base_command,
            *[str(file.relative_to(work_dir)) for file in result_files],
        ]

    def get_command_create_report(self, base_command: list[str], result_files: list[Path], work_dir: Path) -> list[str]:
        return [
            *base_command,
            "--report",
            self.report_file,
            *[str(file.relative_to(work_dir)) for file in result_files],
        ]

    @contextlib.contextmanager
    def _set_env_vars(self) -> None:
        """Temporarily sets the configured environment variables for the duration of the context."""
        old_env = os.environ.copy()
        os.environ.update(self.env_variables or {})
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    @staticmethod
    def _args_to_shell_command(args):
        escaped_args = [shlex.quote(arg) for arg in args]
        return " ".join(escaped_args)
