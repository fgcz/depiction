from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from snakemake_invoke.config import ExecutionModel, SnakemakeInvokeConfig
from snakemake_invoke.invoke.invoke_call_function import InvokeCallFunction
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
            InvokeSubprocess(config=self.config).invoke(
                work_dir=work_dir, result_files=result_files
            )
        elif self.config.execution_model == ExecutionModel.CALL_FUNCTION:
            InvokeCallFunction(config=self.config).invoke(
                work_dir=work_dir, result_files=result_files
            )
        else:
            typing.assert_never(self.config.execution_model)

    def dry_run(self, work_dir: Path, result_files: list[Path]) -> None:
        InvokeSubprocess(config=self.config).invoke(
            work_dir, result_files, extra_args=["--dryrun", "--printshellcmds"]
        )
