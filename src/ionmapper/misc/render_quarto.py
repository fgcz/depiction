import os
import shutil
import subprocess
from pathlib import Path

import yaml


class RenderQuarto:
    # This doesn't really work if you need input files in the same directory as the document
    # @classmethod
    # def render_result(cls, document: Path, output_file: Path, parameters: dict[str, str] | None = None):
    #    with tempfile.TemporaryDirectory() as tmp_dir:
    #        result = cls.render(
    #            document=document,
    #            output_dir=tmp_dir,
    #            parameters=parameters,
    #            output_format=output_file.suffix[1:],
    #        )
    #        shutil.move(result, output_file)

    @classmethod
    def render(
        cls,
        document: Path | str,
        output_dir: Path | str,
        parameters: dict[str, str] | None = None,
        output_format: str = "html",
        delete_qmd: bool = False,
    ) -> Path:
        document = Path(document)
        output_dir = Path(output_dir)

        # copy the quarto document to the output directory
        copy_path = output_dir / document.name
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(document, copy_path)

        parameters = parameters or {}
        if parameters:
            config_path = cls._dump_parameters_yaml(parameters=parameters, document=document, output_dir=output_dir)
            execute_params = ["--execute-params", config_path]
        else:
            execute_params = []

        # render the document using the CLI interface
        command = [
            "quarto",
            "render",
            str(copy_path.name),
            "--to",
            output_format,
            *execute_params,
        ]
        print(f"Executing: {' '.join(command)}")
        proc = subprocess.Popen(
            command,
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
        )
        failed = False
        while line := proc.stdout.readline():
            print(line)
            if "An error occurred while executing the following" in line or "Quitting from lines" in line:
                failed = True

        proc.wait()
        if proc.returncode != 0 or failed:
            subprocess.run(["quarto", "check"], cwd=output_dir, check=False)
            raise RuntimeError(f"Rendering failed for {document=} ({proc.returncode=}, {failed=})")

        if delete_qmd:
            copy_path.unlink()

        return output_dir / f"{document.stem}.{output_format}"

    @classmethod
    def _dump_parameters_yaml(cls, parameters: dict[str, str], document: Path, output_dir: Path) -> str:
        output_file = output_dir / f"{document.stem}.yaml"
        with open(output_file, "w") as file:
            yaml.dump(parameters, file)
        return output_file.name

    @classmethod
    def _flags_for_parameters(cls, parameters: dict[str, str]) -> list[str]:
        # see https://quarto.org/docs/computations/parameters.html for details
        if parameters is None:
            return []
        flags = []
        for key, value in parameters.items():
            flags.append("-P")
            flags.append(f"{key}:{value}")
        return flags
