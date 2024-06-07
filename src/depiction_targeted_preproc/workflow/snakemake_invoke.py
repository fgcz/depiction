import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class SnakemakeInvoke:
    use_subprocess: bool = True

    def invoke(self, work_dir: Path, result_files: list[Path], n_cores: int = 1) -> None:
        if self.use_subprocess:
            self._invoke_subprocess(work_dir, result_files, n_cores)
        else:
            self._invoke_direct(work_dir, result_files, n_cores)

    @property
    def snakefile_path(self) -> Path:
        return Path(__file__).parents[1] / "workflow" / "experimental.smk"

    @property
    def workflow_dir(self) -> Path:
        return Path(__file__).parents[1] / "workflow"

    def _invoke_direct(self, work_dir: Path, result_files: list[Path], n_cores: int) -> None:
        pass

    def _invoke_subprocess(self, work_dir: Path, result_files: list[Path], n_cores: int) -> None:
        snakemake_bin = shutil.which("snakemake")
        if snakemake_bin is None:
            raise RuntimeError(f"snakemake not found, check PATH: {os.environ['PATH']}")
        command = [
            snakemake_bin,
            "-d",
            str(work_dir),
            "--cores",
            str(n_cores),
            "--snakefile",
            str(self.snakefile_path),
            *[str(file.relative_to(work_dir)) for file in result_files],
        ]
        logger.info("Executing {command}", command=command)
        subprocess.run(
            command,
            cwd=self.workflow_dir,
            check=True,
        )
