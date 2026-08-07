from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExecutionModel(str, Enum):
    CALL_FUNCTION = "call_function"
    SUBPROCESS = "subprocess"


class SnakemakeInvokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snakefile_path: Path = Path("Snakefile").absolute()
    execution_model: ExecutionModel = ExecutionModel.SUBPROCESS
    print_shell_commands: bool = False
    continue_on_error: bool = False
    report_file: str | None = None
    n_cores: int = 1
    env_variables: dict[str, str] | None = None
