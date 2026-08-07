from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from snakemake_invoke.config import SnakemakeInvokeConfig

# TODO this one is not fully on par in terms of features


@dataclass
class InvokeCallFunction:
    config: SnakemakeInvokeConfig

    def invoke(self, work_dir: Path, result_files: list[Path]) -> None:
        from snakemake.api import (
            DAGSettings,
            ExecutionSettings,
            OutputSettings,
            ResourceSettings,
            SnakemakeApi,
            StorageSettings,
        )

        with SnakemakeApi(
            OutputSettings(
                printshellcmds=self.config.print_shell_commands,
                verbose=True,
                show_failed_logs=True,
            ),
        ) as snakemake_api:
            workflow_api = snakemake_api.workflow(
                storage_settings=StorageSettings(),
                resource_settings=ResourceSettings(cores=self.config.n_cores),
                snakefile=self.config.snakefile_path,
                workdir=work_dir,
            )
            dag_api = workflow_api.dag(
                dag_settings=DAGSettings(
                    targets=[str(p) for p in result_files], force_incomplete=True
                )
            )
            with self._set_env_vars():
                dag_api.execute_workflow(
                    execution_settings=ExecutionSettings(
                        keep_going=self.config.continue_on_error
                    )
                )
            # TODO this misses the report file generation

    @contextlib.contextmanager
    def _set_env_vars(self) -> Generator[None, None, None]:
        """Temporarily sets the configured environment variables for the duration of the context."""
        old_env = os.environ.copy()
        os.environ.update(self.config.env_variables or {})
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old_env)
