from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from snakemake_invoke.config import SnakemakeInvokeConfig, ExecutionModel
from snakemake_invoke.invoke.invoke_subprocess import InvokeSubprocess


@dataclass
class SnakemakeInvoke:
    config: SnakemakeInvokeConfig

    def invoke(self, work_dir: Path, result_files: list[Path]) -> None:
        """Invokes the snakemake workflow to generate the requested result files.
        :param work_dir: The working directory where the data folder structure is located.
        :param result_files: The list of result files to generate (relative to `work_dir`).
        """
        if self.config.execution_model == ExecutionModel.SUBPROCESS:
            InvokeSubprocess(config=self.config).invoke(work_dir, result_files)
        elif self.config.execution_model == ExecutionModel.CALL_FUNCTION:
            self._invoke_direct(work_dir, result_files)
        else:
            raise ValueError(f"Unknown execution model: {self.config.execution_model}")

    def dry_run(self, work_dir: Path, result_files: list[Path]) -> None:
        InvokeSubprocess(config=self.config).invoke(work_dir, result_files, extra_args=["--dryrun", "--printshellcmds"])

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
            # TODO this misses the report file generation

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
