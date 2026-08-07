from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from snakemake_invoke.config import SnakemakeInvokeConfig


@dataclass
class InvokeSubprocess:
    config: SnakemakeInvokeConfig

    def invoke(
        self,
        work_dir: Path,
        result_files: list[Path],
        extra_args: list[str] | None = None,
    ) -> None:
        extra_args = extra_args or []
        if self.config.continue_on_error:
            extra_args.append("--keep-going")
        if self.config.print_shell_commands:
            extra_args.append("--printshellcmds")
        base_command = self.get_base_command(extra_args=extra_args, work_dir=work_dir)
        command = self.get_command_create_results(
            base_command=base_command, result_files=result_files, work_dir=work_dir
        )
        self._execute_command(command)
        command = self.get_command_create_report(
            base_command=base_command, result_files=result_files, work_dir=work_dir
        )
        if command:
            self._execute_command(command)

    def _execute_command(self, command: list[str]) -> None:
        logger.info("Executing {command}", command=self._args_to_shell_command(command))
        subprocess.run(
            command,
            cwd=self.config.snakefile_path.parent,
            check=True,
            # TODO sort of redundant with non-subprocess helper?
            env={**os.environ, **(self.config.env_variables or {})},
        )

    def get_command_create_results(
        self, base_command: list[str], result_files: list[Path], work_dir: Path
    ) -> list[str]:
        return [
            *base_command,
            *[str(file.relative_to(work_dir)) for file in result_files],
        ]

    def get_command_create_report(
        self, base_command: list[str], result_files: list[Path], work_dir: Path
    ) -> list[str]:
        if self.config.report_file is None:
            return []
        return [
            *base_command,
            "--report",
            self.config.report_file,
            *[str(file.relative_to(work_dir)) for file in result_files],
        ]

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
            str(self.config.snakefile_path),
            # TODO configurable
            "--rerun-incomplete",
            *extra_args,
        ]

    @staticmethod
    def _args_to_shell_command(args: list[str]) -> str:
        escaped_args = [shlex.quote(arg) for arg in args]
        return " ".join(escaped_args)
