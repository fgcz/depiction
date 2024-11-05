from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from snakemake_invoke.config import SnakemakeInvokeConfig, ExecutionModel


@dataclass
class SnakemakeInvoke:
    config: SnakemakeInvokeConfig

    def invoke(self, work_dir: Path, result_files: list[Path]) -> None:
        """Invokes the snakemake workflow to generate the requested result files.
        :param work_dir: The working directory where the data folder structure is located.
        :param result_files: The list of result files to generate (relative to `work_dir`).
        """
        if self.config.execution_model == ExecutionModel.SUBPROCESS:
            self._invoke_subprocess(work_dir, result_files)
        elif self.config.execution_model == ExecutionModel.CALL_FUNCTION:
            self._invoke_direct(work_dir, result_files)
        else:
            raise ValueError(f"Unknown execution model: {self.config.execution_model}")

    def dry_run(self, work_dir: Path, result_files: list[Path]) -> None:
        self._invoke_subprocess(work_dir, result_files, extra_args=["--dryrun", "--printshellcmds"])

    @property
    def snakefile_path(self) -> Path:
        # TODO this should not be hardcoded, and should be updated
        return Path(__file__).parents[1] / "workflow" / self.config.snakefile_name

    @property
    def workflow_dir(self) -> Path:
        # TODO this should not be hardcoded, and should be updated
        # TODO code duplication
        return Path(__file__).parents[1] / "workflow"

    def _invoke_direct(self, work_dir: Path, result_files: list[Path]) -> None:
        from snakemake.api import (
            SnakemakeApi,
            OutputSettings,
            StorageSettings,
            ResourceSettings,
            DAGSettings,
            ExecutionSettings,
        )

        with SnakemakeApi(
            OutputSettings(
                verbose=True,
                show_failed_logs=True,
            ),
        ) as snakemake_api:
            workflow_api = snakemake_api.workflow(
                storage_settings=StorageSettings(),
                resource_settings=ResourceSettings(cores=self.config.n_cores),
                snakefile=self.snakefile_path,
                workdir=work_dir,
            )
            dag_api = workflow_api.dag(
                dag_settings=DAGSettings(targets=[str(p) for p in result_files], force_incomplete=True)
            )
            with self._set_env_vars():
                dag_api.execute_workflow(execution_settings=ExecutionSettings(keep_going=self.config.continue_on_error))

    def _invoke_subprocess(self, work_dir: Path, result_files: list[Path], extra_args: list[str] | None = None) -> None:
        extra_args = extra_args or []
        if self.config.continue_on_error:
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
            # TODO sort of redundant with non-subprocess helper?
            env={**os.environ, **(self.config.env_variables or {})},
        )
        if self.config.report_file:
            command = self.get_command_create_report(
                base_command=base_command, result_files=result_files, work_dir=work_dir
            )
            logger.info("Executing {command}", command=self._args_to_shell_command(command))
            subprocess.run(
                command,
                cwd=self.workflow_dir,
                check=True,
                env={**os.environ, **(self.config.env_variables or {})},
            )

    def get_base_command(self, extra_args: list[str], work_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "snakemake",
            "-d",
            str(work_dir.absolute()),
            "--cores",
            str(self.config.n_cores),
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
            self.config.report_file,
            *[str(file.relative_to(work_dir)) for file in result_files],
        ]

    @contextlib.contextmanager
    def _set_env_vars(self) -> None:
        """Temporarily sets the configured environment variables for the duration of the context."""
        old_env = os.environ.copy()
        os.environ.update(self.config.env_variables or {})
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    @staticmethod
    def _args_to_shell_command(args):
        escaped_args = [shlex.quote(arg) for arg in args]
        return " ".join(escaped_args)
