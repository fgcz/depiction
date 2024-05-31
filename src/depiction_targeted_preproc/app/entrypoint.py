from pathlib import Path
from typing import Annotated

import typer


def entrypoint(
    script_version: Annotated[str, typer.Option(..., help="The version of the script to run.")] = "",
    input_dir: Annotated[Path, typer.Option(..., help="The input directory.")] = Path("/data/input"),
    output_dir: Annotated[Path, typer.Option(..., help="The output directory.")] = Path("/data/output"),
) -> None:
    if script_version != "":
        raise ValueError(f"Unknown script version: {script_version}")

    work_dir = output_dir / "work"
    work_dir.mkdir(exist_ok=True)

    # TODO for every input we need: imzML file, mass list

    # for input_file in input_dir.glob("*.imzML"):
    #


def main() -> None:
    """Provides the CLI around `entrypoint`."""
    typer.run(entrypoint)


if __name__ == "__main__":
    main()
